from typing import Optional

from source.experiments.semiotic_modelling.content import symbolic_probability, symbolic_predict, SymbolicContent, symbolic_adapt, SYMBOL, SHAPE_A, \
    ACTION, STATE, HISTORY, CONDITION, LEVEL

BASIC_SHAPE = SYMBOL


def symbolic_layer(level: int, model, state: STATE, action: Optional[ACTION], consequence: SHAPE_A, sig: float = .1, alp: float = 1., h: int = 1):
    if level < len(state):
        history = state[level]                  # type: HISTORY
        condition = tuple(history), action      # type: CONDITION

        if level + 1 < len(state):
            upper_history = state[level + 1]            # type: HISTORY
            upper_shape = upper_history[-1]             # type: SHAPE_A
            upper_layer = model[level]                  # type: LEVEL
            upper_content = upper_layer[upper_shape]    # type: SymbolicContent

            if symbolic_probability(upper_content, condition, consequence, alp=alp) < sig:
                if level + 2 < len(state):
                    uppest_layer = model[level + 1]                                                     # type: LEVEL
                    uppest_history = state[level + 2]                                                   # type: HISTORY
                    uppest_shape = uppest_history[-1]                                                   # type: SHAPE_A
                    uppest_content = uppest_layer[uppest_shape]                                         # type: SymbolicContent
                    abstract_condition = tuple(upper_history), condition                                # type: CONDITION
                    upper_shape = symbolic_predict(uppest_content, abstract_condition, default=upper_shape)     # type: SHAPE_A
                    upper_content = upper_layer[upper_shape]                                            # type: SymbolicContent

                    if upper_content is None or symbolic_probability(upper_content, condition, consequence, alp=alp) < sig:
                        upper_content = max(upper_layer.values(), key=lambda _x: symbolic_probability(_x, condition, consequence, alp=alp))  # type: SymbolicContent
                        upper_shape = hash(upper_content)

                        if symbolic_probability(upper_content, condition, consequence, alp=alp) < sig:
                            upper_shape = len(upper_layer)                                # type: SHAPE_A
                            upper_content = SymbolicContent(upper_shape)                          # type: SymbolicContent
                            upper_layer[upper_shape] = upper_content

                else:
                    upper_shape = len(upper_layer)                                        # type: SHAPE_A
                    upper_content = SymbolicContent(upper_shape)                                  # type: SymbolicContent
                    upper_layer[upper_shape] = upper_content

                symbolic_layer(level + 1, model, state, condition, upper_shape)

        else:
            upper_shape = 0                             # type: SHAPE_A
            upper_content = SymbolicContent(upper_shape)        # type: SymbolicContent
            upper_history = [upper_shape]               # type: HISTORY
            state.append(upper_history)
            upper_layer = {upper_shape: upper_content}  # type: LEVEL
            model.append(upper_layer)

        # TODO: externalise to enable parallelisation. change this name to "change state"
        # and perform adaptation afterwards from copy of old state + action to new state
        symbolic_adapt(upper_content, condition, consequence)

    elif level == 0:
        history = []               # type: HISTORY
        state.append(history)

    else:
        raise ValueError("Level too high.")

    history = state[level]                              # type: HISTORY
    history.append(consequence)
    while h < len(history):
        history.pop(0)
