import email.parser
import html2text

def extract_email_text(email_file=None, msg=None):
    parser = email.parser.Parser()
    if email_file is not None:
        msg = parser.parse(email_file)
    if msg is None:
        raise Exception("extract_text: Should provide email_file or msg!")
    return extract_message_text(msg, False)

def extract_message_text(message, parse_html=True):
    subject = message['subject']
    payload = message.get_payload()
    # TODO: parse different transfer-encoding types
    if type(payload) == str:
        content = payload
    if type(payload) == list:
        sub_contents = []
        for m in payload:
            sub_contents.append(extract_message_text(m))
        content = " ".join(sub_contents)
    if parse_html:
        content = html2text.html2text(content)
    if subject is None:
        return content
    return "{0} {1}".format(subject, content)


def test():
    with open('spamassasin/spam/0001.bfc8d64d12b325ff385cca8d07b84288',
    'r') as f:
        text = extract_email_text(email_file=f)
        print(text)

if __name__ == '__main__':
    test()
