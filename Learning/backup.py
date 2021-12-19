class CNN(nn.Module):
    """Classifing neural network that determines if a game will be liked

   
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.learning_rate = 0.005 
        self.hidden_size = hidden_size
        self.hidden = torch.nn.init.xavier_uniform_(torch.zeros(1, hidden_size,dtype=torch.float32), gain=1.0)
        
        self.i2hlin = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h = nn.Sigmoid()
        self.i2o = nn.Linear(hidden_size, output_size)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.loss_fn   = torch.nn.CrossEntropyLoss()
        

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        self.hidden = self.i2hlin(combined.float())
        self.hidden = self.i2h(self.hidden.float())
        output      = self.i2o(self.hidden.float())
        
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
   

    def train(self,labels, games,epoch):
        running_loss = 0.0
        last_loss = 0.0
        for i in range(len(games)):
            self.optimizer.zero_grad()
            outputs = self(torch.from_numpy(games[i:i+1]))

            loss = self.loss_fn(outputs, labels[i:i+1])
            
            loss.backward()

            self.optimizer.step()

            #Gather data and report
            running_loss += loss.item()
            loss = None
            

        return running_loss

       
    def randomChoice(self,l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self):
        game = self.randomChoice(games)

        game_tensor = torch.from_numpy(game)
        output = torch.ones(1,1)
        return output,game_tensor
