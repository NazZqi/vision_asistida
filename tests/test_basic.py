import unittest
# Importamos la función que SÍ existe desde tu script
from src.vision_asistida.vision_asistida import join_spanish

class TestJoinSpanish(unittest.TestCase):

    def test_lista_vacia(self):
        """Prueba que una lista vacía devuelva un string vacío."""
        self.assertEqual(join_spanish([]), "")

    def test_un_elemento(self):
        """Prueba que un solo elemento se devuelva tal cual."""
        self.assertEqual(join_spanish(["izquierda"]), "izquierda")

    def test_dos_elementos(self):
        """Prueba que dos elementos se unan con 'y'."""
        self.assertEqual(join_spanish(["izquierda", "derecha"]), "izquierda y derecha")

    def test_tres_elementos(self):
        """Prueba que tres o más elementos usen comas y 'y'."""
        self.assertEqual(
            join_spanish(["izquierda", "al frente", "derecha"]),
            "izquierda, al frente y derecha"
        )

if __name__ == '__main__':
    unittest.main()