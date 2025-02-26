import random
import json
import openai
import ast
import time
from multiprocessing.pool import Pool
from pycocotools.coco import COCO
from evaluation_script.cocoeval_mp import COCOevalMP

def evaluate(test_annotation_file, user_submission_file, phase_codename, api_key, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    json_gt = json.load(open(test_annotation_file))[:5]
    json_pred = json.load(open(user_submission_file))[:5]

    data_dict = {}

    for item in json_gt:
        unique_key = f"{item['VideoID']}_{item['Q']}"
        if unique_key  not in data_dict:
            data_dict[unique_key] = {'a': item['A'], 'q': item['Q']}

    for item in json_pred:
        unique_key = f"{item['VideoID']}_{item['Q']}"
        data_dict[unique_key] = {'pred': item['A']}

    # Set the OpenAI API key.
    openai.api_key = api_key
    result_qa_pair = annotate(data_dict)

    # v3det_gt = COCO(test_annotation_file)  # gt annotation file
    # v3det_dt = v3det_gt.loadRes(user_submission_file)  # coco-format det results
    # v3det_eval = COCOevalMP(v3det_gt, v3det_dt, 'bbox', num_proc=8)
    # v3det_eval.params.maxDets = [300]

    # v3det_eval.evaluate()
    # v3det_eval.accumulate()
    # v3det_eval.summarize()


    # output = {}
    # if phase_codename == "dev":
    #     print("Evaluating for Dev Phase")
    #     output["result"] = [
    #         {
    #             "OVD": {
    #                 "bAP": random.randint(0, 99),
    #                 "nAP": random.randint(0, 99),
    #                 "AP": random.randint(0, 99),
    #             }
    #         }
    #     ]
    #     print("Completed evaluation for Dev Phase")
    # elif phase_codename == "test":
    #     print("Evaluating for Test Phase")
    #     output["result"] = [
    #         {
    #             "split": "train_split",
    #             "show_to_participant": True,
    #             "accuracies": {"Metric1": 90},
    #         },
    #         {
    #             "split": "test_split",
    #             "show_to_participant": False,
    #             "accuracies": {"Metric1": 50, "Metric2": 40},
    #         },
    #     ]
    #     print("Completed evaluation for Test Phase")
    return result_qa_pair



def annotate(prediction_set):
    result_qa_pair = []
    for ind, (k , single_dict) in enumerate(prediction_set.items()):
        if ind > 10:
            break
        question = single_dict['q']
        answer = single_dict['a']
        pred = single_dict['pred']
        while True:
            try:
                # Compute the correctness score
                completion = openai.ChatCompletion.create(
                    model="GPT-3.5-turbo-0125",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
                                "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
                                "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n"
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Ground truth correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                                "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is  a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
                                "Only provide the Python dictionary string."
                                "For example, your response should look like this: {'pred': 'correct', 'score': 4.8, 'reason': reason}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                try:
                    response_dict = ast.literal_eval(response_message)
                except:
                    # Remove the special characters.
                    start_index = response_message.find("'reason': '") + len("'reason': '")
                    end_index = response_message.find("'", start_index)

                    # Extract the reason value
                    reason_value = response_message[start_index:end_index]

                    # Remove single quotes from the reason value
                    reason_value = reason_value.replace("'", "")

                    # Replace the original reason value with the modified one
                    response_message = response_message[:start_index] + reason_value + '\'}'
                    response_dict = ast.literal_eval(response_message)
                result_qa_pair.append([response_dict, single_dict])
                break
            except Exception as e:
                print(f"Error processing ': {e}")
                time.sleep(60)

    return result_qa_pair
