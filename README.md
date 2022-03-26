# Microbiome-based-disease-prediction-with-prototypical-network
  
 
 ### run prototypcial network

 ```python
 
    from net.PrototypicalNet import PrototypicalNet
    import torch.nn as nn
    from torch import optim
    
    pn = PrototypicalNet(X_train.shape[1], 2, embedding_num, support_num, query_num)    
    pn_optimer = optim.Adam(pn.parameters(), lr=0.001, weight_decay=0.001)
    pn_criterion = nn.CrossEntropyLoss()
    pn.fit(X_train, y_train, pn_optimer, pn_criterion, EPOCH=100)
    pre_y, prob_y = pn.predict(X_test)
```

### run prototypcial network with mixed loss function

 ```python
 
    from net.PrototypicalNet import PrototypicalNet
    import torch.nn as nn
    from torch import optim
    
    pn = PrototypicalNet(X_train.shape[1], 2, embedding_num, support_num, query_num)
    pn_optimer = optim.Adam(pn.parameters(), lr=0.001, weight_decay=0.001)
    pn_criterion = PNTripletloss(l=l,margin=margin)
    pn.fit(X_train, y_train, pn_optimer, pn_criterion, EPOCH=100)
    pre_y, prob_y = pn.predict(X_test)
```

### run Adaptive multimodal prototypcial network(AMPN)

 ```python
    from net.AM1 import Adaptive_Cross_Modal_PN
    import torch.nn as nn
    from torch import optim
 
    acmp = Adaptive_Cross_Modal_PN(X1_train.shape[1], X2_train.shape[1], 2, 
                                  embedding_num, support_num, query_num)  #X1_trian is main modal，X2_train is auxiliary modal
    optimer = optim.Adam(acmp.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    acmp.fit(X1_train, X2_train, y_train, optimer, criterion, 100)
    pre_y, prob_y = acmp.predict(X1_test)
```
