import pickle
import sys
import re
import os

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

from poi_email_addresses import poiEmails 

poi_emails = poiEmails()


