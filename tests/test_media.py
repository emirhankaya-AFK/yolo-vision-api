import pytest

from app.services.media import MediaValidationError, decode_image


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(MediaValidationError):
        decode_image(b"not-an-image")
