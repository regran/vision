import base64

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

#authentication
credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials)

with open('book.jpg', 'rb') as image:
    img = base64.b64encode(image.read()) #encode the binary for JSON
    #JSON for wrapper for HTTP POST request for text features
    service_request = service.images().annotate(body={
        'requests':[
            {
                'image':{
                    'content': img.decode('UTF-8')
                },
            'features':[{
                'type': 'TEXT_DETECTION',
                'maxResults': 12
             }
             ]
            }
        ]
    })
    #execute the request and get a response to process
    response = service_request.execute()
    text = response['responses'][0]['textAnnotations'][0]
    print("The text language is {} and it says {}".format(text['locale'], text['description']))
    print(text)
