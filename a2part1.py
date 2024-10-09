#Do not import any other package
import torch

def question1(t, v):
    """
    Define a function to return a tensor with the same shape as t where all elements are v.
    """
    return torch.full(t.shape, v)

def question2(x):
    """
    Define a function to get the second last column from a 2-dimensional tensor.
    """
    return x[:,-2]

def question3(x):
    """
    Define a function to multiply the second half of x by -1
    x's shape is (n, 2d), where n and d are integers. 
    Do not use any loops.
    """
    return torch.cat((x[:,:x.shape[1]//2], -x[:,x.shape[1]//2:]), dim=1)

def question4(t):
    """
    Define a function to concatenate tensor t to the transpose of t along the last dimension
    """
    return torch.cat((t, t.transpose(-2, -1)), dim=-1)

def question5(data):
    """
    Define a function to combine a list of 1-D tensors with different lengths into a new tensor by truncating the longer tensors
    """

    return torch.stack([d[:min([len(x) for x in data])] for d in data])

def question6(x, q, k):
    """
    Define a function that calculates (x*q) * (x*k)^T
    x's shape is (m, d), while q's shape and k's shape are (d, n)
    DO NOT use loop, list comprehension, or any other similar operations.
    """
    x = x.float()
    q = q.float()
    k = k.float()
    return torch.mm(torch.mm(x, q), torch.mm(x, k).T)

def question7(x, q, k):
    """
    Define a function that calculates batched (x_i*q) * (x_i*k)^T
    x's shape is (b, n, m), while q's shape and k's shape are (m, d)
    DO NOT use loop, list comprehension, or any other similar operations.
    """
    x = x.float()
    q = q.float()
    k = k.float()
    return torch.matmul(torch.matmul(x, q), torch.matmul(x, k).transpose(-1, -2))

def question8(x, v):
    """
    Given a 3-D tensor x and padding value v, set any value v in x to 0.
    """
    return torch.where(x == v, torch.tensor(0), x)

def question9(x):
    """
    Given a 3-D tensor x with padding value -1, calculate the average along the second dimension while ignoring the padding value.
    """ 
    return torch.where(x == -1, torch.tensor(0), x).sum(dim=1) / (x != -1).sum(dim=1)
def question10(pairs):
    """
    Define a function to calculate the sum of the squared loss between each pair of vectors.
    """
    return sum([sum((torch.tensor(a) - torch.tensor(b))**2) for a, b in pairs])


def main():
    q1_t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    q1_v = 10
    print('Q1 example input \nshape: {}\nv: {}\n'.format(q1_t, q1_v))
    q1 = question1(q1_t, q1_v)
    print('Q1 example output: \n{}\n'.format(q1))
    q2_input = torch.tensor([[1,2,3,4], [5,6,7,8]])
    print('Q2 example input: \n{}\n'.format(q2_input))
    q2 = question2(q2_input)
    print('Q2 example output: \n{}\n'.format(q2))
    q3_input = torch.tensor([[1,2,3,4], [5,6,7,8]])
    print('Q3 example input: \nx: {}\n'.format(q3_input))
    q3 = question3(q3_input)
    print('Q3 example output: \n{}\n'.format(q3))
    q4_input = torch.tensor([[0,1],[0,2]])
    print('Q4 example input: \n{}\n'.format(q4_input))
    q4 = question4(q4_input)
    print('Q4 example output: \n{}\n'.format(q4))
    q5_input = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
    print('Q5 example input: \n{}\n'.format(q5_input))
    q5 = question5(q5_input)
    print('Q5 example output: \n{}\n'.format(q5))
    q6_input = torch.tensor([[1,2,3], [4,5,6]])
    q6_q = torch.tensor([[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]])
    q6_k = torch.tensor([[0.02, -0.03], [0.03, 0.02], [0.02, 0]])
    print('Q6 example input \nx: {}\nq: {}\nk: {}\n'.format(q6_input, q6_q, q6_k))
    q6 = question6(q6_input, q6_q, q6_k)
    print('Q6 example output: \n{}\n'.format(q6))
    print('Q7 example input \nx: {}\nq: {}\nk: {}\n'.format(torch.tensor([[[1,2,3], [4,5,6]], [[2,1,2], [3,2,3]]]),  q6_q, q6_k))
    q7 = question7(torch.tensor([[[1,2,3], [4,5,6]], [[2,1,2], [3,2,3]]]),  q6_q, q6_k)
    print('Q7 example output: \n{}\n'.format(q7))
    q8_x = torch.tensor([[[1,2], [-1,-1], [-1,-1]], [[2, 2], [2, 1], [-1,-1]], [[3, 4], [1, 3], [3, 6]]])
    q8_v = -1
    print('Q8 example input: \nx: {}\nv: {}\n'.format(q8_x, q8_v))
    q8 = question8(q8_x, q8_v)
    print('Q8 example output: \n{}\n'.format(q8))
    print('Q9 example input: \n{}\n'.format(q8_x))
    q9 = question9(q8_x)
    print('Q9 example output: \n{}\n'.format(q9))
    q10_input = [([1, 1, 1], [2, 2, 2]), ([1, 2, 3], [3, 2, 1]), ([0.1, 0.2, 0.3], [0.33, 0.25, 0.1])]
    print('Q10 example input: \n{}\n'.format(q10_input))
    q10 = question10(q10_input)
    print('Q10 example output: \n', q10, '\n', sep='')
    
    print('\n==== A2 Part 1 Done ====')


if __name__ == "__main__":
    main()
