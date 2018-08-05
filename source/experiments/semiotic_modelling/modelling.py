from typing import TypeVar, Tuple, Union, List, Dict, Optional, Type, Callable

from source.experiments.semiotic_modelling.content import Content, SymbolicContent

TIME = TypeVar("TIME")
BASIC_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_OUT = TypeVar("BASIC_SHAPE_OUT")

EXAMPLE = Tuple[BASIC_IN, BASIC_OUT]
ABSTRACT_SHAPE = int                                        # TODO: make generic hashable
APPEARANCE = Union[BASIC_IN, BASIC_OUT, ABSTRACT_SHAPE]
HISTORY = Union[List[APPEARANCE], Tuple[APPEARANCE, ...]]
LEVEL = Dict[APPEARANCE, Content]
MODEL = List[LEVEL]
STATE = List[APPEARANCE]
TRACE = List[HISTORY]


def predict(model: MODEL, state: STATE, input_value: BASIC_IN) -> Optional[BASIC_OUT]:
    no_model_layers, no_state_layers = len(model), len(state)
    assert no_model_layers >= no_state_layers
    if no_state_layers < 1:
        return None
    content_shape = state[0]
    if no_model_layers < 1:
        return None
    base_layer = model[0]
    content = base_layer.get(content_shape)
    assert content is not None
    return content.predict(input_value, default=None)


def get_content(model: MODEL, state: STATE, level: int) -> Content:
    assert level < len(model)
    layer = model[level]        # type: LEVEL

    assert level < len(state)
    shape = state[level]    # type: APPEARANCE

    content = layer.get(shape)  # type: Content
    assert content is not None
    return content


def update_traces(traces: Tuple[TRACE, ...], states: Tuple[STATE, ...], history_length: int):
    assert len(traces) == len(states)

    for each_trace, each_state in zip(traces, states):
        no_trace_layers, no_state_layers = len(each_trace), len(each_state)
        assert no_trace_layers == no_state_layers

        for each_shape, each_trace_layer in zip(each_state, each_trace):
            each_trace_layer.append(each_shape)
            while history_length < len(each_trace_layer):
                each_trace_layer.pop(0)


def generate_content(model: MODEL, states: Tuple[STATE, ...], base_content: Type[Content], alpha: Callable[[int, int], float]):
    no_model_layers = len(model)
    len_set = set(len(_x) for _x in states)
    assert len(len_set) == 1
    no_state_layers, = len_set
    assert no_model_layers + 1 >= no_state_layers >= no_model_layers
    for _i in range(no_state_layers):
        state_layer_indices_with_new_content = [_j for _j, each_state_layer in enumerate(states) if each_state_layer[_i] == -1]
        if 0 < len(state_layer_indices_with_new_content):
            if _i == no_model_layers:
                each_model_layer = dict()
                model.append(each_model_layer)
            else:
                each_model_layer = model[_i]
            new_shape = len(each_model_layer)
            alpha_value = alpha(_i, new_shape)
            each_model_layer[new_shape] = base_content(new_shape, alpha_value) if _i < 1 else SymbolicContent(new_shape, alpha_value)
            for each_index in state_layer_indices_with_new_content:
                each_state = states[each_index]
                each_state[_i] = new_shape


simple = True


def adapt_abstract_content(model: MODEL, traces: Tuple[TRACE, ...], states: Tuple[STATE, ...]):
    no_traces, no_states = len(traces), len(states)
    assert no_traces == no_states

    no_model_layers = len(model)
    for each_trace, each_state in zip(traces, states):
        no_trace_layers, no_state_layers = len(each_trace), len(each_state)
        assert no_model_layers == no_trace_layers == no_state_layers

        for _i in range(no_state_layers - 1):
            content = get_content(model, each_state, _i + 1)
            shape_out = each_state[_i]

            if _i == 0 or simple:
                abstract_shape = tuple(each_trace[_i])
            else:
                abstract_shape = tuple(each_trace[_i]), tuple(each_trace[_i - 1])          # TODO: keep shape of content? see below

            content.adapt(abstract_shape, shape_out)


def update_situation(shape: BASIC_IN, target_value: BASIC_OUT, model: MODEL, trace: TRACE, state: STATE,
                     sigma: Callable[[int, int], float], fix_at: Callable[[int], int]):
    no_model_layers = len(model)
    level = 0                                                                                                   # type: int

    # for level, each_shape in enumerate(situation):
    while level < no_model_layers:
        layer = model[level]
        no_representations = len(layer)

        s = sigma(level, no_representations)
        content = get_content(model, state, level)                                                          # type: Content
        if content.probability(shape, target_value) >= s:
            break

        if level == 0 or simple:
            abstract_shape = tuple(trace[level])
        else:
            abstract_shape = tuple(trace[level]), shape                     # TODO: keep shape of content? see above
        if level + 1 < no_model_layers:
            context = get_content(model, state, level + 1)                                                      # type: Content
            abstract_target = context.predict(abstract_shape)                                                                      # type: APPEARANCE
            if abstract_target is not None:
                content = layer[abstract_target]                                                                              # type: Content
                if content.probability(shape, target_value) >= s:
                    state[level] = abstract_target
                    target_value = abstract_target
                    shape = abstract_shape
                    level += 1
                    continue

        content = max(layer.values(), key=lambda _x: _x.probability(shape, target_value))               # type: Content
        abstract_target = hash(content)                                                                           # type: APPEARANCE
        if content.probability(shape, target_value) >= s or no_representations >= fix_at(level):
            state[level] = abstract_target
            target_value = abstract_target
            shape = abstract_shape                                                                           # type: HISTORY
            level += 1
            continue

        state[level] = -1                                                                           # type: APPEARANCE
        level += 1


def generate_situation_layer(model: MODEL, states: Tuple[STATE, ...]):
    len_set = {len(each_state) for each_state in states}
    assert len(len_set) == 1
    no_state_layers, = len_set
    assert no_state_layers == len(model)

    if -1 in {each_state[-1] for each_state in states} and len(model[-1]) == 1:
        for each_state in states:
            each_state.append(-1)
