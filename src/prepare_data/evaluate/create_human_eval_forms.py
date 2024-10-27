from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import os.path

SCOPES = ["https://www.googleapis.com/auth/forms.body"]
DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = Flow.from_client_secrets_file(
            'client_secrets.json', SCOPES)
        flow.run_local_server(port=0)
        creds = flow.credentials

    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

form_service = build('forms',
                     'v1',
                     credentials=creds,
                     discoveryServiceUrl=DISCOVERY_DOC)

# Function to create a form with a custom task number


def add_multiple_choice_question(title, options, index):
    NEW_QUESTION = {
        "requests": [
            {
                "createItem": {
                    "item": {
                        "title": title,
                        "questionItem": {
                            "question": {
                                "required": True,
                                "choiceQuestion": {
                                    "type": "RADIO",
                                    "options": options,
                                    "shuffle": True,
                                },
                            }
                        },
                    },
                    "location": {"index": index},
                },
            }
        ]
    }

    return NEW_QUESTION

    # Adds the question to the form


def create_indie_value_reasoning_form(task_number):
    # Request body for creating a form
    NEW_FORM = {
        "info": {
            "title": f"Individualistic Human Values Reasoning Task {task_number}",
            "documentTitle": f"Individualistic Human Values Reasoning Task {task_number}",
        }
    }

    # Request body to add a multiple-choice question
    NEW_QUESTION = {
        "requests": [
            # {
            #     "createItem": {
            #         "item": {
            #             "title": "IMPORTANT (PLEASE READ CAREFULLY)---Demonstration Value-Expressing Statements:",
            #             "textItem": {}
            #         },
            #         "location": {"index": 0},
            #     },
            # },
            # {
            #     "createItem": {
            #         "item": {
            #             "title": (
            #                 "In what year did the United States land a mission on"
            #                 " the moon?"
            #             ),
            #             "questionItem": {
            #                 "question": {
            #                     "required": True,
            #                     "choiceQuestion": {
            #                         "type": "RADIO",
            #                         "options": [
            #                             {"value": "1965"},
            #                             {"value": "1967"},
            #                             {"value": "1969"},
            #                             {"value": "1971"},
            #                         ],
            #                         "shuffle": True,
            #                     },
            #                 }
            #             },
            #         },
            #         "location": {"index": 1},
            #     },
            # },
            # {
            #     "createItem": {
            #         "item": {
            #             "title": (
            #                 "In what year did the United States land a mission on"
            #                 " the moon?"
            #             ),
            #             "questionItem": {
            #                 "question": {
            #                     "required": True,
            #                     "choiceQuestion": {
            #                         "type": "RADIO",
            #                         "options": [
            #                             {"value": "1965"},
            #                             {"value": "1967"},
            #                             {"value": "1969"},
            #                             {"value": "1971"},
            #                         ],
            #                         "shuffle": True,
            #                     },
            #                 }
            #             },
            #         },
            #         "location": {"index": 2},
            #     },
            # },
        ]
    }

    # Creates the initial form
    result = form_service.forms().create(body=NEW_FORM).execute()

    print(f"Form created with ID: {result['formId']}")

    # Update form info to include a description
    UPDATE_FORM_INFO = {
        "requests": [
            {
                "updateFormInfo": {
                    "info": {
                        "description": "Your task is to read a list of value-expressing statements from Person A and based on which to predict Person A's choice among a set of new value-expressing statements. \n\nInstructions:\n\nYou will be provided with a list of *demonstration value-expressing statements* from Person A that reflect Person A's human values and preferences. Your task is to read carefully of these statements to understand Person A's underlying human value system and use this understanding to predict their *most likely choice among new groups of value-expressing statements*. \n\nPlease read very carefully of the demonstration statements before making your predictions."
                    },
                    "updateMask": "description"
                }
            }
        ]
    }

    # Update the form with the description
    form_service.forms().batchUpdate(
        formId=result["formId"], body=UPDATE_FORM_INFO).execute()

    # Adds the question to the form
    question_setting = (
        form_service.forms()
        .batchUpdate(formId=result["formId"], body=NEW_QUESTION)
        .execute()
    )

    # Prints the result to show the question has been added
    get_result = form_service.forms().get(formId=result["formId"]).execute()
    print(f"Form details: {get_result}")

    return result["formId"]


# Example usage
task_number = 1  # You can change this to create forms for different tasks
form_id = create_indie_value_reasoning_form(task_number)
