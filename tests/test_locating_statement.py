from types import SimpleNamespace
import pytest
from algorithm.search.setup import _resolve_sam_prompt
from algorithm.search.config import default_search_config,validate_search_config


def test_statement_preserves_full_target_and_is_physically_counted():
    question='What color is the cargo bed of the pickup diagonally above and to the left of the turquoise car on the bottom approach?'
    statement='Locate the cargo bed of the pickup diagonally above and to the left of the turquoise car on the bottom approach.'
    def rewrite(text,*,on_request):
        assert text==question
        on_request();return statement
    def forbidden(*a,**kw):pytest.fail('Category/owner resolver must not override full statement')
    settings=default_search_config();settings.update(detector_prompt_mode='locating_statement',structured_query=True)
    usage={};events=[]
    _resolve_sam_prompt(settings,'the cargo bed',SimpleNamespace(locate_statement=rewrite,resolve_structured=forbidden),usage,lambda *a:events.append(a),question)
    assert settings['sam_prompt']==statement and settings['detector_statement']==statement
    assert usage['physical_generations']==1 and usage['generation_counts']['resolver']==1


def test_invalid_statement_is_not_silently_replaced_by_category():
    settings=default_search_config();settings['detector_prompt_mode']='locating_statement'
    with pytest.raises(ValueError,match='Invalid locating statement'):
        _resolve_sam_prompt(settings,'car',SimpleNamespace(locate_statement=lambda *a,**kw:'car'),{},lambda *a:None,'What color?')


def test_llmdet_rejects_old_part_path_and_invalid_threshold():
    settings=default_search_config();settings.update(primary_detector='llmdet',part_refinement=True,structured_query=True)
    with pytest.raises(ValueError,match='part refinement'):validate_search_config(settings)
    settings.update(part_refinement=False,llmdet_threshold=2)
    with pytest.raises(ValueError,match='threshold'):validate_search_config(settings)
