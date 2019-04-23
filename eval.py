from pyrouge import Rouge155

r = Rouge155('./pyrouge_master/tools/ROUGE-1.5.5/')
r.model_dir = './Gold_Summaries/'
r.model_filename_pattern = 'Gold.A.#ID#.txt'

r.system_dir = './Model_Summaries/'
r.system_filename_pattern = 'Model.(\d+).txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)