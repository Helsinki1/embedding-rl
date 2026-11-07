"""Separate the two fixed feature-coordinate interventions."""


def transform_component(features, neural, fitted, mode):
    centered = neural.double() - neural.double().mean(1, keepdim=True)
    if mode == "remove_only":
        return centered - features.double() @ fitted["projection"]
    if mode == "scale_only":
        return centered / fitted["scale"]
    raise ValueError("Unknown feature intervention")


def original_component_weights(theta, delta, fitted, mode):
    if mode == "remove_only":
        return theta.double() - fitted["projection"] @ delta.double(), delta.double()
    if mode == "scale_only":
        return theta.double(), delta.double() / fitted["scale"]
    raise ValueError("Unknown feature intervention")
