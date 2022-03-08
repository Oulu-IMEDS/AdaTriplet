import torch
from tqdm import tqdm

from common.dist_metric import DistanceMetric
from common.evaluators import Evaluator
from common.methods import MetricLearningMethods


def distance(f_a, f_p, f_n, margin=1, type='cosine'):
    if type == 'cosine':
        no_tripets = f_a.shape[0]
        no_features = f_a.shape[1]

        ap = torch.matmul(f_a.view(no_tripets, 1, no_features),
                          f_p.view(no_tripets, no_features, 1))
        an = torch.matmul(f_a.view(no_tripets, 1, no_features),
                          f_n.view(no_tripets, no_features, 1))
        d = an - ap + margin

    elif type == 'l2':
        d_ap = torch.nn.functional.pairwise_distance(f_p, f_a, p=2)
        d_an = torch.nn.functional.pairwise_distance(f_n, f_a, p=2)
        d = d_ap - d_an + margin
    else:
        raise ValueError(f'Not support diatance type {type}')
    d = torch.squeeze(d)
    return d


def train_loop(cfg, model, loss_func, mining_func, optimizer, train_loader, epoch_id, device):

    if cfg.loss_identity_func == 'LSE':
        loss_id_selected = loss_func['LogSumExpLoss']
    elif cfg.loss_identity_func == 'LB':
        loss_id_selected = loss_func['LowerBoundLoss']
    else:
        raise ValueError(f'Not support this loss {cfg.loss_identity_func}')

    if cfg.method == 'TripletLSE':
        mining_func = mining_func['TripletMargin']
        loss_matching_func = loss_func['LogSumExpLoss']
        loss_id_func = None
    elif cfg.method == 'Triplet':
        mining_func = mining_func['TripletMargin']
        loss_matching_func = loss_func['Triplet']
        loss_id_func = None
    elif cfg.method == 'ArcFace':
        mining_func = mining_func['Angular']
        loss_matching_func = loss_func['ArcFaceLoss']
        loss_id_func = None
    elif cfg.method == 'SoftTriplet':
        mining_func = mining_func['TripletMargin_lib']
        loss_matching_func = loss_func['SoftTriplet']
        loss_id_func = None
    elif cfg.method == 'LiftedStructure':
        mining_func = mining_func['PairMargin']
        loss_matching_func = loss_func['LiftedStructureLoss']
        loss_id_func = None
    elif cfg.method == 'Contrastive':
        mining_func = mining_func['PairMargin']
        loss_matching_func = loss_func['ContrastiveLoss']
        loss_id_func = None
    elif cfg.method == 'SCT':
        mining_func = mining_func['SCT']
        loss_matching_func = loss_func['LogSumExpLoss']
        loss_id_func = None
    elif cfg.method == 'Triplet-AM':
        mining_func = mining_func['TripletMargin_auto']
        loss_matching_func = loss_func['Triplet']
        loss_id_func = None
    elif cfg.method == 'WAT':
        mining_func = mining_func['TripletAdaptive']
        mining_func.set_mode(mode='weakly')
        loss_matching_func = loss_func['Triplet']
        loss_id_func = None
    elif cfg.method == 'Tuplet':
        mining_func = mining_func['PairMargin']
        loss_matching_func = loss_func['TupletLoss']
        loss_id_func = None
    elif cfg.method == 'AdaTriplet-AM':
        mining_func = mining_func['AutoParams']
        loss_matching_func = loss_func['Triplet']
        loss_id_func = loss_id_selected
    elif cfg.method == 'AdaTriplet' or cfg.method == 'ap':
        mining_func = mining_func['TripletMargin']
        loss_matching_func = loss_func['Triplet']
        loss_id_func = loss_id_selected
    else:
        raise ValueError(f'Not support this method {cfg.method}')

    model.train(True)
    n_batches = len(train_loader)
    progress_bar = tqdm(train_loader, total=n_batches, desc=f"Epoch [{epoch_id}][Train]:")

    sum_loss = 0
    sum_triplets = 0
    sum_neg_pairs = 0
    sum_pos_pairs = 0
    for batch_id, batch in enumerate(progress_bar):
        # Get sampled data and transfer them to the correct device
        data = batch['data'].to(device)
        labels = batch['pid'].to(device)
        optimizer.zero_grad()
        embeddings = model(data)

        method = MetricLearningMethods(cfg, mining_func, loss_matching=loss_matching_func, loss_identity=loss_id_func)
        # Save distribution
        if cfg.save_distribution:
            if batch_id == 0:
                an_distribution = mining_func.get_an_dists(mode='total')
                ap_distribution = mining_func.get_ap_dists(mode='total')

        loss = method.calculate_total_loss(embeddings, labels, epoch_id=epoch_id, batch_id=batch_id)

        no_triplets_batch, no_neg_pairs_batch, no_pos_pairs_batch = method.get_no_triplets()
        if ~torch.isnan(loss):
            sum_loss += loss
        sum_triplets += no_triplets_batch
        sum_neg_pairs += no_neg_pairs_batch
        sum_pos_pairs += no_pos_pairs_batch

        loss.backward()
        optimizer.step()

        metrics_display = {'loss': loss.item(), 'no_triplets': no_triplets_batch, 'an_pairs': no_neg_pairs_batch,
                           'ap_pairs': no_pos_pairs_batch}
        progress_bar.set_postfix(metrics_display)
    mean_loss = sum_loss / n_batches
    mean_triplets = sum_triplets / n_batches
    mean_neg_pairs = sum_neg_pairs / n_batches
    mean_pos_pairs = sum_pos_pairs / n_batches
    print(f'Average Loss: {mean_loss}')
    print(f'Average numbers of triplets: {mean_triplets}')
    print(f'Average numbers of negative pairs: {mean_neg_pairs}')
    print(f'Average numbers of positive pairs: {mean_pos_pairs}')

    if cfg.save_distribution:
        return an_distribution, ap_distribution


def eval_loop(cfg, model, optimizer, eval_loader, eval_query, epoch_id, device, save_model=True, model_name=None,
              return_metrics=False):
    # Tell the model we are not training but evaluating it
    model.train(False)  # or model.eval()
    best_ap = float(cfg.vars.best_ap)
    best_cmc = float(cfg.vars.best_cmc)
    n_batches = len(eval_loader)

    with torch.no_grad():
        # Evaluator
        metric = DistanceMetric(algorithm=cfg.eval_dismat_algorithm)
        evaluator = Evaluator(model, epoch_id, device)
        mAP, cmc_0 = evaluator.evaluate(eval_loader, eval_query, eval_query, metric)

        # Store model based on balanced accuracy
        if mAP > best_ap and save_model:
            model_filename = f'{model_name}_{cfg.personal_id}_mAP.pth'
            print(
                f'Improved mean AP from {best_ap} to {mAP}. Saved to {model_filename}...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean AP': mAP}, model_filename)
            cfg.vars.best_ap = str(mAP)

        if cmc_0 > best_cmc and save_model:
            model_filename = f'{model_name}_{cfg.personal_id}_CMC.pth'
            print(
                f'Improved CMC from {best_cmc} to {cmc_0}. Saved to {model_filename}...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean AP': cmc_0}, model_filename)
            cfg.vars.best_cmc = str(cmc_0)

        if return_metrics:
            return mAP, cmc_0


def test_loop(model, test_loader, test_query, test_gallery, device):
    # Tell the model we are not training but evaluating it
    model.train(False)  # or model.eval()
    # Init dictionary to store metrics
    with torch.no_grad():
        # Evaluator
        metric = DistanceMetric(algorithm='euclidean')
        evaluator = Evaluator(model, 0, device)
        distmat, features = evaluator.evaluate(test_loader, test_query, test_gallery, metric,
                                               return_distmat_features=True)
    return distmat, features
