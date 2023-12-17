import numpy as np
from collections import Counter


def entropy(y): #entropy của hàm toàn cục
    hist = np.bincount(y) # số lần xuất hiện
    ps = hist / len(y) # xác suất xuaatst hiện của mỗi giá trị
    return -np.sum([p * np.log2(p) for p in ps if p > 0]) # đo mức độ khoogn chắc chắn của dữ liệu


class Node: #lưu trữ thông tin của 1 node

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): 
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None): #(phân chia mẫu tối thiểu, độ sâu tối đa, số lượng đặc trưng)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y): #fit method với tập huấn luyện và tập nhãn 
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) # số lượng đặc trưng (n_feats) 
        self.root = self._grow_tree(X, y)

    def predict(self, X): #predict method 
        #Duyệt cây
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        #Trồng cây
        n_samples, n_features = X.shape #số mẫu và số đặc trưng
        n_labels = len(np.unique(y)) #tính số lượng nhãn duy nhất trong mảng y

        #Tiêu chí dừng
        if (depth >= self.max_depth #độ sâu max --> tránh overfitting
                or n_labels == 1 #số lượng nhãn khác nhau = 1 --> tất cả các mẫu trong node đều cùng thuộc 1 nhãn
                or n_samples < self.min_samples_split): #số mẫu nhỏ hơn mẫu tối thiểu --> không đủ tạo  ra node con có ý nghĩa
            leaf_value = self._most_common_label(y) #--> nút gốc thành nút lá và có giá trị tính trong hàm most_common_label
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False) #chọn các đặc trưng ngẫu nhiên và không bị trùng lặp

        # tìm kiếm tham lam 
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs) #--> tìm ra tiêu chí tốt nhất để chia nút hiện tại

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) 
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1) #đệ quy _grow_tree để xây dựng cây con bên trái
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right) #tạo một nút quyết định mới và trả về nó. 

    def _best_criteria(self, X, y, feat_idxs): #tìm ra tiêu chí tốt nhất cho việc chia nút (node)
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx] #lấy ra đặc trưng của X tương ứng với chỉ số feat_idx.
            thresholds = np.unique(X_column) #chứa các ngưỡng duy nhất để chia cột X_column
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh #trả về chỉ số đặc trưng và ngưỡng tốt nhất cho việc chia nút 

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh) #các mẫu sẽ đc chia thành nút con bên trái và bên phải sau phân tách.

        if len(left_idxs) == 0 or len(right_idxs) == 0: #nếu số mẫu của tập dữ liệu con ở nút trái(phải) bằng 0
            # --> thông tin thu được sẽ được thiết lập bằng 0 hay quyết định chia nút sẽ k được thực hiện (--> nút trở thành nút lá)
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y) # số lượng mẫu ban  đầu
        n_l, n_r = len(left_idxs), len(right_idxs) # số lượng mẫu ở nốt trái và nốt phải
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # chỉ số của các mẫu trong X_column <= split_thresh --> sau đó được làm phẳng 
        right_idxs = np.argwhere(X_column > split_thresh).flatten() # chỉ số của các mẫu trong X_column > split_thresh. 
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node): #dự đoán nhãn
        if node.is_leaf_node(): #kiểm tra có là lá hay không
            return node.value #nếu có trả về giá trị nút làm kết quả

        if x[node.feature] <= node.threshold: #kiểm tra đặc trưng của x
            return self._traverse_tree(x, node.left) #nếu nhỏ hơn ngưỡng thì đệ quy xét nốt trái
        return self._traverse_tree(x, node.right) # ngược lại xét nốt phải
    

    def _most_common_label(self, y):
        counter = Counter(y) #ếm số lần xuất hiện của từng giá trị và lưu dưới dạng list
        most_common = counter.most_common(1)[0][0]
        #trả về danh sách các cặp (giá trị, số lần xuất hiện) theo thứ tự giảm dần của số lần xuất hiện.
        #do chỉ quan tâm đến phần tử phổ biến nhất --> counter.most_common(1)
        #most_common(1)[0][0]: lấy ra nhãn phổ biến nhất
        return most_common # --> trả về nhãn phổ biến nhất --> gán giá trị cho nút lá 