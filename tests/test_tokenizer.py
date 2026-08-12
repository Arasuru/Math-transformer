from src.data.tokenizer import CharTokenizer


def test_tokenizer_encode_decode():
    tokenizer = CharTokenizer()
    test_string = "123+456=579"

    encoded = tokenizer.encode(test_string)
    decoded = tokenizer.decode(encoded)

    assert decoded == test_string, (
        f"Data lost during encoded-decode cycle. Expected: {test_string}, Got: {decoded}"
    )
