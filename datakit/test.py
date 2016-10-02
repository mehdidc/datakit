
if __name__ == '__main__':
    from cifar import load
    data = load(which='all')
    print(data['train']['X'].shape)
