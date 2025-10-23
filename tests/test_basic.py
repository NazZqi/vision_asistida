import unittest
from src.vision_asistida.vision_asistida import join_spanish

class TestJoinSpanish(unittest.TestCase):

    def test_lista_vacia(self):
        self.assertEqual(join_spanish([]), "")

    def test_un_elemento(self):
        self.assertEqual(join_spanish(["izquierda"]), "izquierda")

    def test_dos_elementos(self):
        self.assertEqual(join_spanish(["izquierda", "derecha"]), "izquierda y derecha")

    def test_tres_elementos(self):
        self.assertEqual(
            join_spanish(["izquierda", "al frente", "derecha"]),
            "izquierda, al frente y derecha"
        )

if __name__ == '__main__':
    unittest.main()