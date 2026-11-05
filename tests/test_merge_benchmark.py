import json,sys
import pytest
from algorithm.tools import benchmark

def test_official_dataset_reaches_model_loading_with_correct_provenance(tmp_path,monkeypatch):
 data=tmp_path/'data';data.mkdir();(data/'annotations.json').write_text(json.dumps({'images':[{'file_name':'029.jpg','qa_annotations':[{'question_id':'029_q0001','question':'Shape?'}]}]}))
 config=tmp_path/'config.json';config.write_text(json.dumps({'partition_mode':'spatial','qwen_model':'Qwen/Qwen3-VL-8B-Instruct','qwen_quantization':'nf4'}));out=tmp_path/'output'
 monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(data),'--output-dir',str(out),'--config-file',str(config)])
 def stop(*args,**kwargs):raise RuntimeError('MODEL_LOADING_REACHED')
 monkeypatch.setattr(benchmark,'load_qwen_scorer',stop)
 with pytest.raises(RuntimeError,match='MODEL_LOADING_REACHED'):benchmark.main()
 p=json.loads((out/'provenance.json').read_text());assert p['vlm']['model']=='Qwen/Qwen3-VL-8B-Instruct';assert p['vlm']['quantization']=='nf4'
 assert 'answer' not in json.loads((out/'selection.json').read_text())[0]
