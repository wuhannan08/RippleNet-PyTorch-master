import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


class RippleNet(nn.Module):
    def __init__(self, args, n_entity, n_relation, n_user):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation, n_user)

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()

        # 加入dropout
        self.dropout = nn.Dropout(p=0.2)

        # 加一个用户嵌入层mycode
        self.user_emb = nn.Embedding(self.n_user, self.dim)

    def _parse_args(self, args, n_entity, n_relation, n_user):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_user = n_user
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.batch_size = args.batch_size

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
        users: torch.LongTensor # 这一个batch的user索引
    ):
        # [batch size, dim]
        # 获取item的嵌入
        item_embeddings = self.entity_emb(items)
        # 以下三个list最终长度为n_hop，每一个元素shape=[batch_size, n_memory, dim]，关系的是[batch size, n_memory, dim, dim]
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        # 得到ripple_set的嵌入
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            # memories_h[i]返回这个batch所有user第i阶的头实体,shape=[batch_size, n_memory]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        # 计算论文中提到的 o_1,o_2,...,o_h，放入o_list中
        o_list, item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings
        )
        # 根据item嵌入和user嵌入（通过o_list得到）计算相似度得分
        # scores是[batch size, 1]， users是[batch_size, 1]，存放的是user_id
        scores = self.predict(item_embeddings, o_list, users)

        # _compute_loss计算损失返回dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)
        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    # 计算论文中提到的 o_1,o_2,...,o_h，放入o_list中
    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1] 扩充维度
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            # item_embeddings = self.transform_matrix(o)
            item_embeddings = self.dropout(self.transform_matrix(o))
        elif self.item_update_mode == "plus_transform":
            # item_embeddings = self.transform_matrix(item_embeddings + o)
            item_embeddings = self.dropout(self.transform_matrix(item_embeddings + o))
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list, users):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # 最终user的嵌入= o + user_emb
        user_embedding = self.user_emb(users) # 获取这一个batch每条数据对应的user嵌入
        y = user_embedding + y

        # [batch_size] item_embeddings和y都是[batch_size, dim]
        scores = (item_embeddings * y).sum(dim=1)   # scores是[batch size, 1]
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t, users):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t, users)
        # scores和labels都是[batch_size, 1]
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def my_evaluate(self, items, labels, memories_h, memories_r, memories_t, users):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t, users)
        # scores和labels都是[batch_size, 1]
        scores = return_dict["scores"].detach().cpu().numpy()

        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        # 计算precision、recall、F1
        ones = np.ones(labels.shape)
        # 预测为正的所有 np.sum(np.equal(ones, predictions))
        # TP：预测为正的正类
        TP = 0
        TP_TN = np.equal(predictions, labels)   # 预测正确的样本
        for i in range(TP_TN.shape[0]): # 遍历预测正确的样本
            if TP_TN[i] == 1 and predictions[i] == 1:
                TP += 1

        precision = TP / np.sum(np.equal(ones, predictions))

        # 预测出了多少真正类
        # 样本中有多少正类np.sum(np.equal(ones, labels))
        recall = TP / np.sum(np.equal(ones, labels))

        F1 = 2 * precision * recall / (precision + recall)

        return auc, acc, precision, recall, F1

    # 构造 user-item-predict矩阵，每一行是user i对所有项目的预测打分
    def update_predict_matrix(self, items, users, scores, user_item_matrix_info):
        user_dict, item_dict = user_item_matrix_info[0], user_item_matrix_info[1]
        # user_item_matrix, user_item_predict = user_item_matrix_info[2], user_item_matrix_info[3]

        for item, user, score in zip(np.array(items), np.array(users), np.array(scores)):
            i = user_dict.get(user)
            j = item_dict.get(item)
            user_item_matrix_info[3][i][j] = score

    def top_k_evaluate(self, items, labels, memories_h, memories_r, memories_t, users, user_item_matrix_info):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t, users)
        # scores和labels都是[batch_size, 1]
        scores = return_dict["scores"].detach().cpu().numpy()

        items = items.cpu().numpy()
        users = users.cpu().numpy()

        # 更新 用户-项目-预测 矩阵
        self.update_predict_matrix(items, users, scores, user_item_matrix_info)