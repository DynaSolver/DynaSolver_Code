from data_provider.data_loader import IPC


def get_data(args, full_mesh=False):
    data_dict = {
        "IPC": IPC,
    }
    if args.loader not in data_dict:
        raise ValueError(f"Unsupported loader '{args.loader}'. IPC-only pipeline supports: {list(data_dict)}")
    dataset = data_dict[args.loader](args)
    train_loader, test_loader, shapelist = dataset.get_loader(full_mesh=full_mesh)
    return dataset, train_loader, test_loader, shapelist
