from common.fl_models.iris.utils import Utils


def create_local_learner():
    import pandas as pd
    import torch
    from flwr.common import MetricsRecord, ParametersRecord
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset

    class SoftmaxModel(nn.Module):
        def __init__(self, input_size: int):
            super(SoftmaxModel, self).__init__()
            self.linear = nn.Linear(input_size, 3)
            self.softmax = nn.Softmax(dim=1)
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        def forward(self, x):
            x = self.linear(x)
            return self.softmax(x)

        def get_parameters(self) -> ParametersRecord:
            return Utils.pytorch_to_parameter_record(self.state_dict())

        def set_parameters(self, parameters: ParametersRecord):
            self.load_state_dict(Utils.parameters_to_pytorch_state_dict(parameters))

        def prepare_data(self, data):
            class IrisDataset(Dataset):
                def __init__(self, dataframe: pd.DataFrame):
                    self.dataframe = dataframe
                    self.dataframe.loc[:, "class"] = dataframe.loc[:, "class"].replace(
                        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
                    )

                def __getitem__(self, idx: int):
                    x = self.dataframe.iloc[idx, :-1].to_numpy("float32")
                    y = torch.tensor(self.dataframe.iloc[idx, -1], dtype=torch.long)
                    return x, y

                def __len__(self):
                    return len(self.dataframe)

            dataset = IrisDataset(data)
            dataloader = DataLoader(dataset, batch_size=32)
            self.dataloader = dataloader

        def train(self):
            for batch in self.dataloader:
                x, y = batch
                self.optimizer.zero_grad()
                y_hat = self.forward(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()

            return MetricsRecord({"loss": loss.item()})

        def evaluate(self):
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in self.dataloader:
                    x, y = batch
                    y_hat = self.forward(x)
                    _, predicted = torch.max(y_hat, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            return MetricsRecord({"accuracy": correct / total})

    return SoftmaxModel(4)


if __name__ == "__main__":
    import pandas as pd

    model = create_local_learner()
    model.prepare_data(pd.read_csv("data/iris.csv"))
    metrics = model.train()
    print(metrics)
    metrics = model.evaluate()
    print(metrics)
