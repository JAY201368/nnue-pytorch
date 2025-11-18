from torch.utils.data import DataLoader

import data_loader
import time


def make_data_loaders(
    train_filenames,
    val_filenames,
    feature_name,
    num_workers,
    batch_size,
    config: data_loader.DataloaderSkipConfig,
    epoch_size,
    val_size,
):
    # Epoch and validation sizes are arbitrary
    features_name = feature_name
    train_infinite = data_loader.SparseBatchDataset(
        features_name,
        train_filenames,
        batch_size,
        num_workers=num_workers,
        config=config,
    )
    val_infinite = data_loader.SparseBatchDataset(
        features_name,
        val_filenames,
        batch_size,
        config=config,
    )
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(
            train_infinite, (epoch_size + batch_size - 1) // batch_size  # 批次数向上取整
        ),
        batch_size=None,
        batch_sampler=None,
    )
    val = DataLoader(
        data_loader.FixedNumBatchesDataset(
            val_infinite, (val_size + batch_size - 1) // batch_size
        ),
        batch_size=None,
        batch_sampler=None,
    )
    return train, val


def test_data_loaders():
    print("=== 开始验证数据加载器 ===")
    train_loader, val_loader = make_data_loaders(
        train_filenames=["data/large_gensfen_multipvdiff_100_d9.binpack"],
        val_filenames=["data/large_gensfen_multipvdiff_100_d9.binpack"],
        feature_name="HalfKAv2_hm^",
        num_workers=1,
        batch_size=16,
        config=data_loader.DataloaderSkipConfig(),
        epoch_size=50,
        val_size=10,
    )

    print("✓ DataLoader创建成功")
    print(f"训练加载器类型: {type(train_loader)}")
    print(f"验证加载器类型: {type(val_loader)}")

    print("\n1. 测试迭代功能...")
    try:
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        print("✓ 迭代器创建成功")
    except Exception as e:
        print(f"❌ 迭代器创建失败: {e}")
        return

    print("\n2. 测试批次获取...")
    try:
        train_batch = next(train_iter)
        print("✓ 训练批次获取成功")

        val_batch = next(val_iter)
        print("✓ 验证批次获取成功")

        return train_loader, val_loader, train_batch, val_batch

    except StopIteration:
        print("⚠️ 数据加载器为空")
    except Exception as e:
        print(f"❌ 批次获取失败: {e}")
        import traceback
        traceback.print_exc()

    return train_loader, val_loader, None, None


def analyze_batch_data(batch, name="批次"):
    print(f"\n=== 分析 {name} ===")

    if batch is None:
        print("❌ 批次为空")
        return

    print(f"批次类型: {type(batch)}")

    if isinstance(batch, (list, tuple)):
        print(f"包含 {len(batch)} 个元素")
        for i, item in enumerate(batch):
            print(f"  元素 {i}: {type(item)}")
            if hasattr(item, 'shape'):
                print(f"      形状: {item.shape}, 类型: {item.dtype}, 设备: {item.device}")
            else:
                print(f"      值: {item}")
    else:
        print(f"未知批次结构: {type(batch)}")
        assert 0


def validate_data_loader(loader, loader_name, num_batches=3):
    print(f"\n=== 验证 {loader_name} ===")
    try:
        iterator = iter(loader)

        for i in range(num_batches):
            print(f"\n批次 {i + 1}:")
            try:
                batch = next(iterator)
                analyze_batch_data(batch, f"{loader_name}批次{i + 1}")

                # 如果是稀疏特征批次，进行详细分析
                if isinstance(batch, (list, tuple)) and len(batch) >= 8:
                    analyze_sparse_features(batch)

            except StopIteration:
                print(f"⚠️ {loader_name} 只有 {i} 个批次")
                break
            except Exception as e:
                print(f"❌ 获取批次 {i + 1} 失败: {e}")
                break

    except Exception as e:
        print(f"❌ 创建 {loader_name} 迭代器失败: {e}")


def analyze_sparse_features(batch):
    print("  稀疏特征分析:")

    # 假设批次是 (us, them, white_indices, white_values, black_indices, black_values, outcome, score, ...)
    if len(batch) >= 8:
        us, them, white_indices, white_values, black_indices, black_values, outcome, score = batch[:8]

        print(f"    白方特征索引: {white_indices.shape}")
        print(f"    白方特征值: {white_values.shape}")
        print(f"    黑方特征索引: {black_indices.shape}")
        print(f"    黑方特征值: {black_values.shape}")
        print(f"    胜负结果: {outcome.shape}")
        print(f"    分数: {score.shape}")

        # # 统计稀疏性
        white_sparsity = (white_indices == -1).float().mean().item()
        black_sparsity = (black_indices == -1).float().mean().item()
        
        print(f"    白方特征稀疏度: {white_sparsity:.1%}")
        print(f"    黑方特征稀疏度: {black_sparsity:.1%}")
        print(f"    平均胜负: {outcome.mean().item():.3f}")
        print(f"    分数范围: [{score.min().item()}, {score.max().item()}]")


if __name__ == '__main__':
    train_loader, val_loader, train_batch, val_batch = test_data_loaders()
    validate_data_loader(train_loader, "训练数据加载器", num_batches=3)
    validate_data_loader(val_loader, "验证数据加载器", num_batches=1)
    time.sleep(10)  # 等一下后台worker退出