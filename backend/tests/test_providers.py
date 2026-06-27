from app.providers import get_chat_model
from app.settings import Settings


def test_chat_model_can_disable_thinking_and_limit_output():
    model = get_chat_model(
        Settings(ZAI_API_KEY="test-key"),
        "glm-5.2",
        thinking=False,
        max_tokens=700,
    )

    assert model.extra_body == {"thinking": {"type": "disabled"}}
    assert model.max_tokens == 700
