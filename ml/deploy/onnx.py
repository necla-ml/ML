
class ONNXPredictor(object):
    def __init__(self, engine):
        self.session = engine

    def predict(self, *args, **kwargs):
        session = self.session
        inputs = session.get_inputs()
        batch = {input.name: arg.detach().cpu().numpy() if arg.requires_grad else arg.cpu().numpy() for input, arg in zip(inputs, args)}
        outputs = session.run(None, batch)
        return outputs