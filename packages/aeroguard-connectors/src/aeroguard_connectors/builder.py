"""Builder backend that lets the TimeNet SDK discover this workspace's connectors.

The SDK never imports a connector package directly: a build backend registers itself under the
``timenet.builders`` entry-point group and the SDK looks it up by dataset id. Registering this class
there (see ``pyproject.toml``) is what makes ``TimeNet(registry=...).load("nasa/cmapss")`` build from
the workspace connector instead of falling back to the upstream ``timenet-connectors`` copy.
"""

from pathlib import Path

from timenet.engine import run_pipeline
from timenet.errors import TimeNetInvalidCardError
from timenet.types import DatasetMetadata

#: Dataset ids this workspace builds, mapped to the directory holding the connector and its card.
CONNECTOR_DIRS: dict[str, Path] = {"nasa/cmapss": Path(__file__).parent / "cmapss"}


def _card(dataset_id: str) -> Path | None:
    """Locate a dataset's card without importing its connector module.

    Args:
        dataset_id: The dataset id.

    Returns:
        The card path, or ``None`` when this workspace has no connector for the id.
    """
    directory = CONNECTOR_DIRS.get(dataset_id)
    if directory is None:
        return None
    card = directory / "dataset.yaml"
    return card if card.is_file() else None


class AeroGuardBuilder:
    """Builds an AeroGuard workspace dataset by running its connector in this interpreter.

    Unlike the upstream ``ConnectorBuilder``, this backend does not re-exec into a per-connector
    virtual environment. The connector's dependencies are the project's own, so the environment
    ``make sync`` produces is already the one the build needs.
    """

    def knows(self, dataset_id: str) -> bool:
        """Report whether this workspace has a connector for the id, without importing it.

        Args:
            dataset_id: The dataset id.

        Returns:
            Whether a connector with a readable card exists for the id.
        """
        return _card(dataset_id) is not None

    def declared_version(self, dataset_id: str) -> str | None:
        """Read the version the connector's card declares, without importing or building it.

        Args:
            dataset_id: The dataset id.

        Returns:
            The declared version string, or ``None`` when no readable card exists.
        """
        card = _card(dataset_id)
        if card is None:
            return None
        try:
            return str(DatasetMetadata.from_yaml(card).dataset_version)
        except TimeNetInvalidCardError:
            return None

    def build(
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        values_backend: str | None = None,
    ) -> Path:
        """Build the dataset into a local registry directory.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already committed.
            values_backend: Storage backend for the values plane, or ``None`` for the connector's.

        Returns:
            The committed version directory.

        Raises:
            LookupError: If this workspace has no connector for the id.
        """
        if not self.knows(dataset_id):
            raise LookupError(
                f"no AeroGuard connector for {dataset_id!r}; known: {', '.join(CONNECTOR_DIRS)}"
            )
        from aeroguard_connectors.cmapss.connector import CMAPSSConnector  # noqa: PLC0415

        return run_pipeline(
            CMAPSSConnector(), Path(root), force=force, values_backend=values_backend
        )
