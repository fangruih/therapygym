from therapygym.therapy.judge import TherapyJudge


def test_extracts_and_normalizes_scores():
    judge = TherapyJudge.__new__(TherapyJudge)
    result = judge._extract_json_from_response(
        '{"cbt_ratings":{"Agenda":{"rating":3,"evidence":"x"}},'
        '"safety_ratings":{"Crisis":{"rating":"no","evidence":"y"}}}'
    )
    assert result["cbt_scores"]["Agenda"] == 0.5
    assert result["safety_scores"]["Crisis"] == 0.0

