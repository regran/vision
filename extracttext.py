"""Basic applications using the Google Vision API"""
import base64, sys
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

#authentication
credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials)

def imagecontent(image): #encode image for processing
    return base64.b64encode(open(image, 'rb').read())


def getfeatures(image): #extract the text from an image
    #JSON for wrapper for HTTP POST request for text features
    service_request = service.images().annotate(body={
        'requests':[
            {
                'image':{
                    'content':imagecontent(image).decode('UTF-8')
                    },
                'features':[
                    {
                        'type': 'TEXT_DETECTION',
                        'maxResults': 12
                    },
                    {
                        'type': 'SAFE_SEARCH_DETECTION'
                    },
                    {
                        'type': 'FACE_DETECTION'
                    }
                    ]
                }
            ]
        })
    #execute request and get a response to process
    response = service_request.execute()
    faceinfo = None
    textinfo = None
    safeinfo = None
    for feature in response['responses'][0]:
        if feature == 'textAnnotations':
            textinfo = response['responses'][0][feature][0]
        elif feature == 'safeSearchAnnotation':
            safeinfo = response['responses'][0][feature]
        elif feature == 'faceAnnotations':
            faceinfo = response['responses'][0][feature][0]
    return textinfo, faceinfo, safeinfo

textinfo, faceinfo, safeinfo = getfeatures(sys.argv[1])
if textinfo is not None:
    print("There is text in language {} that says {}".format(textinfo['locale'], textinfo['description']))
if faceinfo is not None:
    print("Joy: {} \nSorrow: {} \nAnger: {} \nSurprise: {}".format(faceinfo['joyLikelihood'], faceinfo['sorrowLikelihood'], faceinfo['angerLikelihood'], faceinfo['surpriseLikelihood']))
if safeinfo is not None:
    print("Adult content is {} \nSpoof is {} \nMedical content is {} \nViolent content is {}".format(safeinfo['adult'], safeinfo['spoof'], safeinfo['medical'], safeinfo['violence']))


