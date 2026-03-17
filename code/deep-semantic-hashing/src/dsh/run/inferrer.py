from abc import ABC, abstractmethod
import h5py
import os
from typing import Generic, TypeVar


from dsh.config.data import DC
from dsh.config.model import MC
import dsh.config.run
from dsh.data.dataset import DS
from dsh.model.modelbase import TopLevelModelBase
from dsh.utils.adapter import DatasetToModelAdapter
from dsh.utils.eventsystem import EventSystem
from dsh.utils.logger import Logger
from dsh.utils.types import Constants, DSI

M = TypeVar("M", bound=TopLevelModelBase)


class Inferrer(ABC, Generic[M, MC, DS, DC, DSI]):
    def __init__(self, cfg: dsh.config.run.InferenceRunConfig[MC, DC]):
        self.cfg = cfg

    def run(
        self,
        models_to_load: dict[int, str],
        model: M,
    ):
        Logger().info(f"[INF] Start inference.")
        self.dataset_metadata, self.dataset = self._load_dataset(model.get_adapter())

        EventSystem()[Constants.Events.Infer.Start](self)
        for epoch, model_path in models_to_load.items():
            EventSystem()[Constants.Events.Infer.EpochStart](self, epoch)
            # Load saved model checkpoint
            Logger().info(f"[INF] Loading model epoch {epoch} from {model_path}.")
            self._load_state_dict(model, model_path)

            # Create output directory and output file
            self.cfg.env.add_epoch_resolver(lambda: epoch)
            output_path = self.cfg.env.resolve(self.cfg.env.output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as outfile:
                self._infer(model, outfile)
                outfile.flush()
                outfile.close()
            self.cfg.env.remove_epoch_resolver()
            EventSystem()[Constants.Events.Infer.EpochEnd](self, self.dataset_metadata.dataset_name, epoch)
        EventSystem()[Constants.Events.Infer.End](self)

    @abstractmethod
    def _load_dataset(self, adapter: DatasetToModelAdapter) -> tuple[DSI, DS]:
        raise NotImplementedError("Inferrer must implement the load_dataset method")

    @abstractmethod
    def _load_state_dict(self, model: M, path: str) -> None:
        raise NotImplementedError("Inferrer must implement the load_state_dict method")

    @abstractmethod
    def _infer(self, model: M, outfile: h5py.File) -> None:
        raise NotImplementedError("Inferrer must implement the infer method")
