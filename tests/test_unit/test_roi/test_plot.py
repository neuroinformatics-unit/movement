from movement.roi import PolygonOfInterest


def test_plot() -> None:
    cc_shell = [(0, 0), (1, 0), (1, 1), (0, 1)]
    c_shell = [(0, 0), (0, 1), (1, 1), (1, 0)]

    cc_hole = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    c_hole = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]

    c_c = PolygonOfInterest(
        c_shell,
        holes=[c_hole],
    )
    c_cc = PolygonOfInterest(c_shell, holes=[cc_hole])
    cc_c = PolygonOfInterest(cc_shell, holes=[c_hole])
    cc_cc = PolygonOfInterest(cc_shell, holes=[cc_hole])

    for p in [c_c, c_cc, cc_c, cc_cc]:
        fig, ax = p.plot()
        pass
