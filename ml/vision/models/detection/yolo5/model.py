from torch import hub

GITHUB = 'ultralytics/yolov5'

def forward_once(self, x, profile=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if profile:
            import thop
            o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
            t = torch_utils.time_synchronized()
            for _ in range(10):
                _ = m(x)
            dt.append((torch_utils.time_synchronized() - t) * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output

    if profile:
        print('%.1fms total' % sum(dt))
    self.features = [y[i] for i in (20, 15, 10)]
    return x[0]

def yolo5l(pretrained=False, channels=3, classes=80, fuse=True):
    import types
    m = hub.load(GITHUB, 'yolov5l', pretrained, channels, classes)
    m.forward_once = types.MethodType(forward_once, m)
    m.save.append(20)
    fuse and m.fuse()
    return m

def yolo5x(pretrained=False, channels=3, classes=80, fuse=True):
    import types
    m = hub.load(GITHUB, 'yolov5x', pretrained, channels, classes)
    m.forward_once = types.MethodType(forward_once, m)
    m.save.append(20)
    fuse and m.fuse()
    return m