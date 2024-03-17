import numpy as np
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import umap
import pickle
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.neighbors import KNeighborsClassifier
from torch import optim
from torchvision import models

from utils_week3 import mAP, mapk, visualizer_hook


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(512, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x


class Network:
    def __init__(self, config, train_dataset, test_dataset, train_labels) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=True)
        embed = EmbeddingLayer(config["embed_size"])
        model = torch.nn.Sequential(*list(model.children())[:-1], embed)
        self.model = model.to(self.device)
        if config["arch_type"] == "siamese":
            self.loss_funcs = {"metric_loss": losses.ContrastiveLoss()}
            self.mining_funcs = {"tuple_miner": miners.PairMarginMiner()}
        elif config["arch_type"] == "triplet":  # Triplet loss
            self.loss_funcs = {"metric_loss": losses.TripletMarginLoss(margin=0.1)}
            self.mining_funcs = {"tuple_miner": miners.BatchHardMiner()}
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_labels = train_labels
        self.setup_trainer()

    def setup_trainer(self):
        class_sampler = samplers.MPerClassSampler(
            labels=self.train_labels,
            m=self.config["batch_size"] // 8,
            batch_size=self.config["batch_size"],
            length_before_new_iter=len(self.train_dataset),
        )
        optimizer = optim.Adam(self.model.parameters(), 3e-4)

        record_keeper, _, _ = logging_presets.get_record_keeper(
            self.config["out_path"] + "logs", self.config["out_path"] + "tensorboard"
        )
        dataset_dict = {"val": self.test_dataset}
        model_folder = self.config["out_path"] + "models"

        hooks = logging_presets.get_hook_container(record_keeper)

        # Create the tester
        tester = testers.GlobalEmbeddingSpaceTester(
            end_of_testing_hook=hooks.end_of_testing_hook,
            visualizer=umap.UMAP(),
            visualizer_hook=visualizer_hook,
            dataloader_num_workers=1,
            accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
        )

        end_of_epoch_hook = hooks.end_of_epoch_hook(
            tester, dataset_dict, model_folder, test_interval=10, patience=1
        )

        metric_trainer = trainers.MetricLossOnly(
            models={"trunk": self.model},
            optimizers={"trunk_optimizer": optimizer},
            batch_size=self.config["batch_size"],
            loss_funcs=self.loss_funcs,
            mining_funcs=self.mining_funcs,
            dataset=self.train_dataset,
            data_device=self.device,
            sampler=class_sampler,
            lr_schedulers={
                "trunk_scheduler_by_epoch": optim.lr_scheduler.StepLR(
                    optimizer, step_size=2, gamma=0.9
                )
            },
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook,
        )

        self.metric_trainer = metric_trainer

    def get_neighbors_labels(
        self, query_data, catalogue_labels, catalogue_data, catalogue_meta
    ):
        knn = KNeighborsClassifier(n_neighbors=len(catalogue_labels), p=1)
        knn = knn.fit(catalogue_data, catalogue_labels)
        neighbors = knn.kneighbors(query_data)[1]

        neighbors_labels = []
        for i in range(len(neighbors)):
            neighbors_class = [catalogue_meta[j][1] for j in neighbors[i]]
            neighbors_labels.append(neighbors_class)

        return neighbors_labels

    def train(self, epochs, save_path=None):
        self.metric_trainer.train(1, epochs)
        if save_path:
            torch.save(
                self.model.state_dict(),
                save_path,
            )

    def test(self, load_data: bool = False):
        catalogue_meta = [(x[0].split("/")[-1], x[1]) for x in self.train_dataset.imgs]
        query_meta = [(x[0].split("/")[-1], x[1]) for x in self.test_dataset.imgs]

        if load_data:
            with open(f"pickles/catalogue_data_{self.config['arch_type']}.pkl", "rb") as f:
                catalogue_data = pickle.load(f)
        else:
            catalogue_data = np.empty(
                (len(self.train_dataset), self.config["embed_size"])
            )
            with torch.no_grad():
                self.model.eval()
                for ii, (img, _) in enumerate(self.train_dataset):
                    img = img.to(self.device)
                    catalogue_data[ii, :] = (
                        self.model(img.unsqueeze(0)).squeeze().cpu().numpy()
                    )

            with open(f"pickles/catalogue_data_{self.config['arch_type']}.pkl", "wb") as f:
                pickle.dump(catalogue_data, f)

        if load_data:
            with open(f"pickles/query_data_{self.config['arch_type']}.pkl", "rb") as f:
                query_data = pickle.load(f)
        else:
            query_data = np.empty((len(self.test_dataset), self.config["embed_size"]))
            with torch.no_grad():
                self.model.eval()
                for ii, (img, _) in enumerate(self.test_dataset):
                    img = img.to(self.device)
                    query_data[ii, :] = (
                        self.model(img.unsqueeze(0)).squeeze().cpu().numpy()
                    )

            with open(f"pickles/query_data_{self.config['arch_type']}.pkl", "wb") as f:
                pickle.dump(query_data, f)

        catalogue_labels = np.asarray([x[1] for x in catalogue_meta])
        query_labels = np.asarray([x[1] for x in query_meta])
        query_labels = [x[1] for x in query_meta]

        neighbors_labels = self.get_neighbors_labels(
            query_data, catalogue_labels, catalogue_data, catalogue_meta
        )

        return query_labels, neighbors_labels

    def evaluate(self, query_labels = None, neighbors_labels = None):
        if not query_labels or not neighbors_labels:
            query_labels = neighbors_labels = self.test()
        map1 = mapk(query_labels, neighbors_labels, 1)
        map5 = mapk(query_labels, neighbors_labels, 5)
        map = mAP(query_labels, neighbors_labels)
        return map1, map5, map

    def load_model(self):
        self.model = torch.load(self.config["load_path"])
