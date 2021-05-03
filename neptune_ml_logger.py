import neptune.new as neptune


class NeptuneLogger:

    def __init__(self, project: str, api_token: str, tags: list,
                 parameters: dict, source_files: list, **kwargs) -> None:
        self.project = project
        self.api_token = api_token
        self.tags = tags
        self.parameters = parameters
        self.run = neptune.init(project=project, api_token=api_token,
                                source_files=source_files)
        self.run['sys/tags'].add(tags)
        self.run['parameters'] = parameters

    def log_metrics(self, metrics: dict) -> None:
        """

        :param metrics:
        :return:
        """
        for key, values in metrics.items():
            for val in values:
                self.run[key].log(val)

    def log_q_table(self, path_file: str) -> None:
        """

        :param path_file:
        :return:
        """
        self.run['train/q_table'].upload(path_file)
