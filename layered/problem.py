import sys
import os
import yaml


SCHEMA = os.path.join(os.path.dirname(__file__), 'problem.yaml')


class Problem:

    def __init__(self, definition=None, schema=SCHEMA):
        self.schema = {'type': 'dict', 'content': self._load_yaml(schema)}
        definition = self._load_yaml(definition)
        self.__dict__.update(self._parse(definition, self.schema))
        self._check_invariants()

    def _check_invariants(self):
        # TODO: Write proper tests and remove this conditional.
        if not self.layers:
            return
        data_in = len(self.dataset.training[0].data)
        data_out = len(self.dataset.training[0].target)
        layer_in = self.layers[0].size
        layer_out = self.layers[-1].size
        assert layer_in == data_in, (
            'the size of the input layer {} must match the data {}.'
            .format(layer_in, data_in))
        assert layer_out == data_out, (
            'the size of the output layer {} must match the labels {}.'
            .format(layer_out, data_out))

    def _parse(self, value, schema):
        """
        Recursively parse a definition against a schema. This includes
        importing modules and looking for types in there.
        """
        if isinstance(value, dict):
            default = schema.get('default', {}).copy()
            default.update(value)
            value = default
            return self._parse_object(value, schema)
        elif isinstance(value, list):
            return self._parse_list(value, schema)
        elif isinstance(value, str):
            return self._parse_object({'type': value}, schema)
        else:
            return self._parse_value(value, schema)

    def _parse_object(self, value, schema):
        Type = self._parse_type(schema, value)
        if 'type' in value:
            del value['type']
        kwargs = schema.get('content', {}).copy()
        kwargs.update(value)
        kwargs = {k: self._parse(v, schema.get('content', {}).get(k, {}))
                  for k, v in kwargs.items()}
        return self._instantiate(Type, **kwargs)

    def _parse_list(self, value, schema):
        Type = self._parse_type(schema)
        elements = [self._parse(x, schema['content']) for x in value]
        return Type(elements)

    def _parse_value(self, value, schema):
        if 'type' not in schema:
            return value
        Type = self._parse_type(schema)
        return self._instantiate(Type, value)

    def _parse_type(self, schema, value=None):
        scopes = [__builtins__]
        # Import source module if provided.
        module = schema.get('module', None)
        if module:
            __import__(module)
            scopes.insert(0, sys.modules[module])
        # Find base type.
        assert 'type' in schema, 'Each property in the schema must have a type'
        Base = self._find_type(schema['type'], scopes)
        # Try to find and validate inherited type.
        Type = Base
        if isinstance(value, dict) and 'type' in value:
            name = value['type'] if 'type' in value else value
            Type = self._find_type(name, scopes)
            assert issubclass(Type, Base), (
                'Expected type compatible to {} but got {}.'
                .format(Base.__name__, Type.__name__))
        return Type

    @staticmethod
    def _find_type(name, scopes):
        name = str(name)
        for scope in scopes:
            if isinstance(scope, dict) and name in scope:
                return scope[name]
            if hasattr(scope, name):
                return getattr(scope, name)
        raise NameError('Could not find type ' + name)

    @staticmethod
    def _instantiate(Type, *args, **kwargs):
        try:
            return Type(*args, **kwargs)
        except TypeError as e:
            print('Wrong parameters for {}.'.format(Type.__name__))
            raise e

    @staticmethod
    def _load_yaml(source):
        """Load a YAML file or string."""
        if source and os.path.isfile(source):
            with open(source) as file_:
                return yaml.load(file_)
        return yaml.load(source)
