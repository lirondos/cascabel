import unittest

from cascabel.shape import word_shape

SPANISH_TOKENS = [
    "casa", "perro", "gato", "comida", "mesa", "silla", "ventana", "puerta",
    "camino", "camisa", "cabeza", "corazón", "tiempo", "trabajo", "palabra",
    "momento", "manera", "persona", "mundo", "vida", "parte", "lugar",
    "empresa", "caso", "grupo", "problema", "mano", "tarde", "noche", "agua",
] * 3  # frequency-weight a little


class WordShapeTestCase(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(word_shape("Hello"), "Xxx")
        self.assertEqual(word_shape("HELLO"), "XX")
        self.assertEqual(word_shape("covid19"), "xxdd")
        self.assertEqual(word_shape("e-mail"), "x-xx")
        self.assertEqual(word_shape("iPhone"), "xXxx")
        self.assertEqual(word_shape(""), "")

    def test_max_run(self):
        self.assertEqual(word_shape("Hello", max_run=4), "Xxxxx")
        self.assertEqual(word_shape("aaaa", max_run=1), "x")


if __name__ == "__main__":
    unittest.main()
