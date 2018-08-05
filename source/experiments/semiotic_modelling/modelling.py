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
SITUATION = List[APPEARANCE]
STATE = List[HISTORY]


def predict(model: MODEL, situation: SITUATION, input_value: BASIC_IN) -> Optional[BASIC_OUT]:
    len_model, len_situation = len(model), len(situation)
    assert len_model >= len_situation
    if len_situation < 1:
        return None
    content_shape = situation[0]
    if len_model < 1:
        return None
    base_layer = model[0]
    content = base_layer.get(content_shape)
    assert content is not None
    return content.predict(input_value, default=None)


def get_content(model: MODEL, situation: SITUATION, level: int) -> Content:
    assert level < len(model)
    layer = model[level]        # type: LEVEL

    assert level < len(situation)
    shape = situation[level]    # type: APPEARANCE

    content = layer.get(shape)  # type: Content
    assert content is not None
    return content


def update_states(states: Tuple[STATE, ...], situations: Tuple[SITUATION, ...], history_length: int):
    assert len(states) == len(situations)

    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        assert len_state == len_situation

        for each_shape, each_layer in zip(each_situation, each_state):
            each_layer.append(each_shape)
            while history_length < len(each_layer):
                each_layer.pop(0)


def generate_content(model: MODEL, situations: Tuple[SITUATION, ...], base_content: Type[Content], alpha: Callable[[int, int], float]):
    len_model = len(model)
    len_set = set(len(_x) for _x in situations)
    assert len(len_set) == 1
    len_situation, = len_set
    assert len_situation == len_model or len_situation == len_model + 1
    for _i in range(len_situation):
        situations_with_new_content = [_j for _j, each_situation in enumerate(situations) if each_situation[_i] == -1]
        if 0 < len(situations_with_new_content):
            if _i == len_model:
                each_layer = dict()
                model.append(each_layer)
            else:
                each_layer = model[_i]
            new_shape = len(each_layer)
            alpha_value = alpha(_i, new_shape)
            each_layer[new_shape] = base_content(new_shape, alpha_value) if _i < 1 else SymbolicContent(new_shape, alpha_value)
            for each_index in situations_with_new_content:
                each_situation = situations[each_index]
                each_situation[_i] = new_shape


simple = False


def adapt_content(model: MODEL, states: Tuple[STATE, ...], situations: Tuple[SITUATION, ...]):
    len_states, len_situations = len(states), len(situations)
    assert len_states == len_situations

    len_model = len(model)
    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        assert len_model == len_state == len_situation

        for _i in range(len_situation - 1):
            content = get_content(model, each_situation, _i + 1)
            shape_out = each_situation[_i]

            if _i == 0 or simple:
                abstract_shape = tuple(each_state[_i])
            else:
                abstract_shape = tuple(each_state[_i]), tuple(each_state[_i - 1])          # TODO: keep shape of content? see below

            content.adapt(abstract_shape, shape_out)


def update_situation(shape: BASIC_IN, target_value: BASIC_OUT,
                     model: MODEL, state: STATE, situation: SITUATION,
                     sigma: Callable[[int, int], float], fix_at: Callable[[int], int]):
    len_model = len(model)
    level = 0                                                                                                   # type: int

    # for level, each_shape in enumerate(situation):
    while level < len_model:
        layer = model[level]
        len_layer = len(layer)

        s = sigma(level, len_layer)
        content = get_content(model, situation, level)                                                          # type: Content
        if content.probability(shape, target_value) >= s:
            break

        if level == 0 or simple:
            abstract_shape = tuple(state[level])
        else:
            abstract_shape = tuple(state[level]), shape                     # TODO: keep shape of content? see above
        if level + 1 < len_model:
            context = get_content(model, situation, level + 1)                                                      # type: Content
            abstract_target = context.predict(abstract_shape)                                                                      # type: APPEARANCE
            if abstract_target is not None:
                content = layer[abstract_target]                                                                              # type: Content
                if content.probability(shape, target_value) >= s:
                    situation[level] = abstract_target
                    target_value = abstract_target
                    shape = abstract_shape
                    level += 1
                    continue

        content = max(layer.values(), key=lambda _x: _x.probability(shape, target_value))               # type: Content
        abstract_target = hash(content)                                                                           # type: APPEARANCE
        if content.probability(shape, target_value) >= s or len_layer >= fix_at(level):
            situation[level] = abstract_target
            target_value = abstract_target
            shape = abstract_shape                                                                           # type: HISTORY
            level += 1
            continue

        situation[level] = -1                                                                           # type: APPEARANCE
        level += 1


def generate_layer(model: MODEL, situations: Tuple[SITUATION, ...]):
    len_set = {len(each_situation) for each_situation in situations}
    assert len(len_set) == 1
    len_situation, = len_set
    assert len_situation == len(model)

    if -1 in {each_situation[-1] for each_situation in situations} and len(model[-1]) == 1:
        for each_situation in situations:
            each_situation.append(-1)
