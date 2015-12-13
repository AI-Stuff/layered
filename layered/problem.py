import os
import yaml
import layered.cost
import layered.dataset
import layered.activation
from layered.network import Layer


class Problem:

    def __init__(self, content=None):
        """
        Construct a problem. If content is specified, try to load it as a YAML
        path and otherwise treat it as an inline YAML string.
        """
        if content and os.path.isfile(content):
            with open(content) as file_:
                self.parse(file_)
        elif content:
            self.parse(content)
        self._validate()

    def __str__(self):
        keys = self.__dict__.keys() & self._defaults().keys()
        return str({x: getattr(self, x) for x in keys})

    def parse(self, definition):
        definition = yaml.load(definition)
        self._load_definition(definition)
        self._load_symbols()
        self._load_layers()
        self._load_weight_tying()
        assert not definition, (
            'unknown properties {} in problem definition'
            .format(', '.join(definition.keys())))

    def _load_definition(self, definition):
        # The empty dictionary causes defaults to be loaded even if the
        # definition is None.
        if not definition:
            definition = {}
        for name, default in self._defaults().items():
            type_ = type(default)
            self.__dict__[name] = type_(definition.pop(name, default))

    def _load_symbols(self):
        # pylint: disable=attribute-defined-outside-init
        self.cost = self._find_symbol(layered.cost, self.cost)()
        self.dataset = self._find_symbol(layered.dataset, self.dataset)()

    def _load_layers(self):
        for index, layer in enumerate(self.layers):
            size, activation = layer.pop('size'), layer.pop('activation')
            activation = self._find_symbol(layered.activation, activation)
            self.layers[index] = Layer(size, activation)

    def _load_weight_tying(self):
        # pylint: disable=attribute-defined-outside-init
        self.weight_tying = [[y.split(',') for y in x]
                             for x in self.weight_tying]
        for i, group in enumerate(self.weight_tying):
            for j, slices in enumerate(group):
                for k, slice_ in enumerate(slices):
                    slice_ = [int(s) if s else None for s in slice_.split(':')]
                    self.weight_tying[i][j][k] = slice(*slice_)
        for i, group in enumerate(self.weight_tying):
            for j, slices in enumerate(group):
                assert not slices[0].start and not slices[0].step, (
                    'Ranges are not allowed in the first dimension.')
                self.weight_tying[i][j][0] = slices[0].stop

    def _find_symbol(self, module, name, fallback=None):
        """
        Find the symbol of the specified name inside the module or raise an
        exception.
        """
        if not hasattr(module, name) and fallback:
            return self._find_symbol(module, fallback, None)
        return getattr(module, name)

    def _validate(self):
        num_input = len(self.dataset.training[0].data)
        num_output = len(self.dataset.training[0].target)
        if self.layers:
            assert self.layers[0].size == num_input, (
                'the size of the input layer must match the training data')
            assert self.layers[-1].size == num_output, (
                'the size of the output layer must match the training labels')

    @staticmethod
    def _defaults():
        return {
            'cost': 'SquaredError',
            'dataset': 'Modulo',
            'layers': [],
            'epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.1,
            'momentum': 0.0,
            'weight_scale': 0.1,
            'weight_decay': 0.0,
            'weight_tying': [],
            'evaluate_every': 1000,
        }
