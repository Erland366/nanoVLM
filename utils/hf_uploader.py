import time
import logging
from concurrent.futures import Future
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


class AsyncHFUploader:
    """Background HuggingFace checkpoint uploader using built-in async API."""

    def __init__(
        self,
        repo_id: str | None = None,
        repo_name_base: str = "nanoVLM",
        max_pending: int = 3,
        max_retries: int = 3,
    ):
        self.max_pending = max_pending
        self.max_retries = max_retries
        self.api = HfApi()

        # Resolve repo_id: use provided or construct from repo_name_base + timestamp
        self.repo_id = self._resolve_repo_id(repo_id, repo_name_base)

        # Validate authentication and repo access at startup
        self._validate_auth_and_repo()

        # Track futures: (step, local_path, attempt_count, future)
        self._pending_futures: list[tuple[int, str, int, Future]] = []
        self._retry_queue: list[tuple[int, str, int]] = []  # (step, path, attempts)
        self._completed_steps: list[int] = []
        self._failed_steps: list[int] = []

    def _resolve_repo_id(self, repo_id: str | None, repo_name_base: str) -> str:
        """Resolve repo_id from explicit value or construct from repo_name_base + timestamp."""
        if repo_id:
            return repo_id

        # Need to get username to construct repo_id
        try:
            user_info = self.api.whoami()
            username = user_info.get("name")
            if not username:
                raise RuntimeError("Could not determine HuggingFace username")
        except Exception as e:
            raise RuntimeError(
                f"HuggingFace authentication failed. Please run 'huggingface-cli login' "
                f"or set the HF_TOKEN environment variable. Error: {e}"
            )

        # Construct repo name: {base}_{timestamp}
        timestamp = time.strftime("%m%d-%H%M%S")
        repo_name = f"{repo_name_base}_{timestamp}"
        return f"{username}/{repo_name}"

    def _validate_auth_and_repo(self):
        """Validate HuggingFace authentication and repo write access."""
        # Check if user is logged in
        try:
            user_info = self.api.whoami()
            username = user_info.get("name", user_info.get("fullname", "unknown"))
            logger.info(f"Authenticated as HuggingFace user: {username}")
        except Exception as e:
            raise RuntimeError(
                f"HuggingFace authentication failed. Please run 'huggingface-cli login' "
                f"or set the HF_TOKEN environment variable. Error: {e}"
            )

        # Check if repo exists, create if not
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            logger.info(f"Found existing repo: {self.repo_id}")
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                # Repo doesn't exist, try to create it
                logger.info(f"Repo {self.repo_id} not found, creating...")
                try:
                    self.api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True)
                    logger.info(f"Created repo: {self.repo_id}")
                except Exception as create_err:
                    raise RuntimeError(
                        f"Failed to create repo {self.repo_id}. "
                        f"Check that you have permission to create repos under this namespace. "
                        f"Error: {create_err}"
                    )
            elif e.response.status_code == 403:
                raise RuntimeError(
                    f"Access denied to repo {self.repo_id}. "
                    f"Check that you have write permission to this repo."
                )
            else:
                raise RuntimeError(f"Failed to access repo {self.repo_id}: {e}")

    def submit_upload(self, local_path: str, step: int):
        """Submit checkpoint for async upload. Drops oldest if too many pending."""
        # Process retries first, then cleanup
        self._process_retries()
        self._cleanup_completed()

        # Drop oldest pending if exceeds max
        while len(self._pending_futures) >= self.max_pending:
            dropped = self._pending_futures.pop(0)
            dropped[3].cancel()  # Cancel the future
            logger.warning(f"Dropping oldest pending upload: step_{dropped[0]}")

        # Submit new upload
        self._submit_async(local_path, step, attempt=0)

    def _submit_async(self, local_path: str, step: int, attempt: int):
        """Internal: submit async upload."""
        branch_name = f"step-{step}"

        # Create branch (synchronous, but fast)
        try:
            self.api.create_branch(repo_id=self.repo_id, branch=branch_name)
        except Exception:
            pass  # Branch may already exist

        # Submit async upload using built-in run_as_future
        future = self.api.upload_folder(
            folder_path=local_path,
            repo_id=self.repo_id,
            revision=branch_name,
            commit_message=f"Checkpoint at step {step}",
            run_as_future=True,  # Non-blocking!
        )

        self._pending_futures.append((step, local_path, attempt, future))
        logger.info(f"Submitted async upload for step_{step} (attempt {attempt + 1})")

    def _cleanup_completed(self):
        """Check completed futures, queue failures for retry."""
        still_pending = []
        for step, local_path, attempt, future in self._pending_futures:
            if future.done():
                try:
                    future.result()  # Raises if failed
                    self._completed_steps.append(step)
                    logger.info(f"Upload completed for step_{step}")
                except Exception as e:
                    if attempt + 1 < self.max_retries:
                        # Queue for retry
                        self._retry_queue.append((step, local_path, attempt + 1))
                        logger.warning(
                            f"Upload failed for step_{step} (attempt {attempt + 1}): {e}. "
                            f"Queued for retry."
                        )
                    else:
                        self._failed_steps.append(step)
                        logger.error(
                            f"Upload failed for step_{step} after {self.max_retries} attempts: {e}"
                        )
            else:
                still_pending.append((step, local_path, attempt, future))
        self._pending_futures = still_pending

    def _process_retries(self):
        """Resubmit queued retries."""
        retries = self._retry_queue[:]
        self._retry_queue = []
        for step, local_path, attempt in retries:
            if len(self._pending_futures) < self.max_pending:
                self._submit_async(local_path, step, attempt)
            else:
                # Still too many pending, re-queue
                self._retry_queue.append((step, local_path, attempt))

    def shutdown(self, wait: bool = True, timeout: float = 300.0):
        """Wait for all pending uploads to complete."""
        if not wait:
            return

        logger.info(f"Waiting for {len(self._pending_futures)} pending uploads...")
        deadline = time.time() + timeout

        for step, local_path, attempt, future in self._pending_futures:
            remaining = max(0, deadline - time.time())
            try:
                future.result(timeout=remaining)
                self._completed_steps.append(step)
                logger.info(f"Upload completed for step_{step}")
            except Exception as e:
                self._failed_steps.append(step)
                logger.error(f"Upload failed for step_{step}: {e}")

        self._pending_futures = []

    def get_status(self) -> dict:
        """Return upload status for logging."""
        self._cleanup_completed()
        return {
            "pending": len(self._pending_futures),
            "retrying": len(self._retry_queue),
            "completed": len(self._completed_steps),
            "failed": len(self._failed_steps),
        }
