import json
import unittest

from algorithm.core.calls import charge_generation, generate_rating, invoke_counted
from algorithm.core.errors import LocalBudgetExceeded


class GenerationAccountingTests(unittest.TestCase):
    def test_repairs_cannot_cross_physical_budget(self):
        usage = {}
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return 'invalid JSON'

        with self.assertRaises(LocalBudgetExceeded):
            generate_rating(generate, json.loads, 'prompt',
                            lambda: charge_generation(usage, 'verification', 1), 'repair')
        self.assertEqual(len(prompts), 1)
        self.assertEqual(usage['physical_generations'], 1)

    def test_successful_repair_is_a_second_generation(self):
        usage = {}
        replies = iter(['broken', '{"answer":"Yes"}'])
        result, raw, prompt = generate_rating(lambda p: next(replies), json.loads, 'initial',
            lambda: charge_generation(usage, 'verification', 2), 'repair')
        self.assertEqual(result['answer'], 'Yes')
        self.assertEqual(usage['search_generations'], 2)
        self.assertEqual(usage['generation_counts'], {'verification': 2})
        self.assertIn('repair', prompt)

    def test_resolver_and_answer_are_counted_separately(self):
        usage = {}
        charge_generation(usage, 'resolver')
        charge_generation(usage, 'verification', 1)
        charge_generation(usage, 'answer')
        self.assertEqual(usage['physical_generations'], 3)
        self.assertEqual(usage['search_generations'], 1)
        self.assertEqual(usage['local_score_calls'], 2)
        with self.assertRaises(LocalBudgetExceeded):
            charge_generation(usage, 'verification', 1)

    def test_native_callback_and_legacy_callable_each_charge_once(self):
        calls = []
        def native(value, *, on_request):
            on_request()
            return value
        self.assertEqual(invoke_counted(native, 3, on_request=lambda: calls.append(1)), 3)
        self.assertEqual(invoke_counted(lambda v: v, 4, on_request=lambda: calls.append(1)), 4)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
