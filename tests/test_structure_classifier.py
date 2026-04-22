from analysis.structure_analyzer import _classify_section


def test_classify_intro_and_outro_by_position_and_energy() -> None:
    intro = _classify_section(
        idx=0,
        n_segments=6,
        start=0.0,
        end=12.0,
        duration=120.0,
        energy=0.25,
        brightness=0.30,
        bass=0.20,
        energy_high=0.70,
        energy_low=0.35,
        prev_energy=None,
        next_energy=0.50,
    )
    outro = _classify_section(
        idx=5,
        n_segments=6,
        start=105.0,
        end=120.0,
        duration=120.0,
        energy=0.20,
        brightness=0.20,
        bass=0.10,
        energy_high=0.70,
        energy_low=0.35,
        prev_energy=0.50,
        next_energy=None,
    )

    assert intro == "intro"
    assert outro == "outro"


def test_classify_high_energy_sections_as_chorus_or_drop() -> None:
    chorus = _classify_section(
        idx=2,
        n_segments=6,
        start=35.0,
        end=50.0,
        duration=120.0,
        energy=0.82,
        brightness=0.55,
        bass=0.40,
        energy_high=0.75,
        energy_low=0.35,
        prev_energy=0.50,
        next_energy=0.60,
    )
    drop = _classify_section(
        idx=3,
        n_segments=6,
        start=50.0,
        end=64.0,
        duration=120.0,
        energy=0.84,
        brightness=0.40,
        bass=0.62,
        energy_high=0.75,
        energy_low=0.35,
        prev_energy=0.70,
        next_energy=0.55,
    )

    assert chorus == "chorus"
    assert drop == "drop"


def test_classify_bridge_in_back_half_after_peak() -> None:
    label = _classify_section(
        idx=4,
        n_segments=7,
        start=70.0,
        end=86.0,
        duration=120.0,
        energy=0.45,
        brightness=0.35,
        bass=0.30,
        energy_high=0.75,
        energy_low=0.35,
        prev_energy=0.80,
        next_energy=0.46,
    )
    assert label == "bridge"
