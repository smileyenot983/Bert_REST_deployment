import torch
import transformers
from models import BertClassifier


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Predictor:
    def __init__(self):
        super().__init__()

        model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'DeepPavlov/rubert-base-cased')
        # using pure bert for embedding extraction
        model = model_class.from_pretrained(pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.bert_clf = BertClassifier(model).to(device)
        model_path = 'bert_pretrained.pt'

        self.bert_clf.load_state_dict(torch.load(model_path,map_location=device))

    def predict(self,sentence):
        sent_tokenized = self.tokenizer(sentence)

        # adding batch dimension
        features = torch.tensor(sent_tokenized['input_ids']).unsqueeze(0).to(device)

        # here we do not have any paddings, thus mask is all ones
        mask = torch.tensor(sent_tokenized['attention_mask']).unsqueeze(0).to(device)

        output = self.bert_clf(features,attention_mask=mask)

        pred_score = float(output)
        pred_class = 'python' if pred_score<0.6 else 'data science'

        return pred_class, pred_score