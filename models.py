from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model, dropout=0.1):
        super().__init__()

        self.bert = pretrained_model
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.cls = nn.Sequential(
            nn.Linear(768,1),
            nn.Sigmoid()
        )

    def forward(self, inputs, attention_mask):
        
        last_hidden_states = self.bert(inputs,attention_mask=attention_mask)
        
        bert_out = self.relu(self.dropout(last_hidden_states[0])) #out = [batch_size,max_len,emb_dim]

        
        
        proba = self.cls(bert_out[:,0,:])[:,0]

        # features = torch.cat([elem[:, 0, :] for elem in features], dim=0).numpy()

        # proba = [batch_size, ] - probability to be positive
        return proba