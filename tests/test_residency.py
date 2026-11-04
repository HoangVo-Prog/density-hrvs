import unittest
from algorithm.models.residency import staged_callable


class ResidencyTests(unittest.TestCase):
    def test_success_and_failure_both_offload_before_next_model(self):
        moves=[]
        wrapped=staged_callable(lambda value:value+1,moves.append,'cpu')
        self.assertEqual(wrapped(4),5);self.assertEqual(moves,['cpu','cpu'])
        def failure():raise ValueError('inference failed')
        wrapped=staged_callable(failure,moves.append,'cpu')
        with self.assertRaises(ValueError):wrapped()
        self.assertEqual(moves,['cpu','cpu','cpu','cpu'])
