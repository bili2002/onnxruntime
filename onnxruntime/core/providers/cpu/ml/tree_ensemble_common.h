// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>

#include "tree_ensemble_aggregator.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "tree_ensemble_helper.h"

namespace onnxruntime {
namespace ml {
namespace detail {

class TreeEnsembleCommonAttributes {
 public:
  int64_t get_target_or_class_count() const { return this->n_targets_or_classes_; }
  virtual Status Init(const OpKernelInfo&) = 0;
  virtual Status compute(OpKernelContext*, const Tensor*, Tensor*, Tensor*) const = 0;
  virtual ~TreeEnsembleCommonAttributes() {}

 protected:
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t n_nodes_;
  int64_t max_tree_depth_;
  int64_t max_feature_id_;
  int64_t n_trees_;
  bool same_mode_;
  bool has_missing_tracks_;
  int parallel_tree_;    // starts parallelizing the computing by trees if n_tree >= parallel_tree_
  int parallel_tree_N_;  // batch size if parallelizing by trees
  int parallel_N_;       // starts parallelizing the computing by rows if n_rows <= parallel_N_
};

// TI: input type
// TH: tree type (types of the node values and targets)
// TO: output type, usually float
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommon : public TreeEnsembleCommonAttributes {
 protected:
  std::vector<ThresholdType> base_values_;
  std::vector<TreeNodeElement<ThresholdType>> nodes_;
  // Type of weights should be a vector of OutputType. Onnx specifications says it must be float.
  // Lightgbm requires a double to do the summation of all trees predictions. That's why
  // `ThresholdType` is used as well for output type (double as well for lightgbm) and not `OutputType`.
  std::vector<SparseValue<ThresholdType>> weights_;
  std::vector<TreeNodeElement<ThresholdType>*> roots_;

  std::vector<double> thresholds;
  std::vector<int64_t> tree_ids;
  std::vector<int64_t> leaves_offset;
  std::vector<uint64_t> bitvectors;

  std::vector<size_t> offsets;
  mutable std::vector<uint64_t> tree_answer;

  struct SplitNode {
    int feature;
    int64_t tree_id;
    double threshold;
    uint64_t bitvector;
  };

  struct SplitNodeCompact {
    int64_t tree_id;
    double threshold;
    uint64_t bitvector;
  };

  std::vector<SplitNodeCompact> split_nodes_;

  std::vector<SplitNode> split_nodes_for_sorting;
  std::vector<ThresholdType> leaves_weight;
  std::vector<std::pair<int32_t&, int32_t&>> leaves_;

  std::vector<size_t> leftLeaves;
  std::vector<size_t> myLeaves;

 public:
  TreeEnsembleCommon() {}

  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_tree_N,
              int parallel_N,
              const std::string& aggregate_function,
              const std::vector<float>& base_values,
              const std::vector<ThresholdType>& base_values_as_tensor,
              int64_t n_targets_or_classes,
              const std::vector<int64_t>& nodes_falsenodeids,
              const std::vector<int64_t>& nodes_featureids,
              const std::vector<float>& nodes_hitrates,
              const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
              const std::vector<int64_t>& nodes_missing_value_tracks_true,
              const std::vector<std::string>& nodes_modes,
              const std::vector<int64_t>& nodes_nodeids,
              const std::vector<int64_t>& nodes_treeids,
              const std::vector<int64_t>& nodes_truenodeids,
              const std::vector<float>& nodes_values,
              const std::vector<ThresholdType>& nodes_values_as_tensor,
              const std::string& post_transform,
              const std::vector<int64_t>& target_class_ids,
              const std::vector<int64_t>& target_class_nodeids,
              const std::vector<int64_t>& target_class_treeids,
              const std::vector<float>& target_class_weights,
              const std::vector<ThresholdType>& target_class_weights_as_tensor);

 protected:
  TreeNodeElement<ThresholdType>* ProcessTreeNodeLeave(TreeNodeElement<ThresholdType>* root,
                                                       const InputType* x_data) const;

  OutputType QuickScorer(const InputType* x_data) const;

  template <typename AGG>
  void ComputeAgg(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Y, Tensor* label, const AGG& agg) const;

 private:
  size_t AddNode(const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
                  const InlinedVector<size_t>& falsenode_ids, const std::vector<int64_t>& nodes_featureids,
                  const std::vector<ThresholdType>& nodes_values_as_tensor, const std::vector<float>& node_values,
                  const std::vector<int64_t>& nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping,
                  int64_t tree_id, const InlinedVector<TreeNodeElementId>& node_tree_ids);
  size_t AddNodes(const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
                  const InlinedVector<size_t>& falsenode_ids, const std::vector<int64_t>& nodes_featureids,
                  const std::vector<ThresholdType>& nodes_values_as_tensor, const std::vector<float>& node_values,
                  const std::vector<int64_t>& nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping,
                  int64_t tree_id, const InlinedVector<TreeNodeElementId>& node_tree_ids);
  void LeftLeaves(const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
                  const InlinedVector<size_t>& falsenode_ids, std::vector<size_t>& used_nodes, size_t left);
};

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  std::vector<ThresholdType> base_values_as_tensor, nodes_hitrates_as_tensor,
      nodes_values_as_tensor, target_weights_as_tensor;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "target_weights_as_tensor", target_weights_as_tensor));
#endif

  return Init(
      80,
      128,
      50,
      info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
      info.GetAttrsOrDefault<float>("base_values"),
      base_values_as_tensor,
      info.GetAttrOrDefault<int64_t>("n_targets", 0),
      info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_featureids"),
      info.GetAttrsOrDefault<float>("nodes_hitrates"),
      nodes_hitrates_as_tensor,
      info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"),
      info.GetAttrsOrDefault<std::string>("nodes_modes"),
      info.GetAttrsOrDefault<int64_t>("nodes_nodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_treeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"),
      info.GetAttrsOrDefault<float>("nodes_values"),
      nodes_values_as_tensor,
      info.GetAttrOrDefault<std::string>("post_transform", "NONE"),
      info.GetAttrsOrDefault<int64_t>("target_ids"),
      info.GetAttrsOrDefault<int64_t>("target_nodeids"),
      info.GetAttrsOrDefault<int64_t>("target_treeids"),
      info.GetAttrsOrDefault<float>("target_weights"),
      target_weights_as_tensor);
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
    int parallel_N,
    const std::string& aggregate_function,
    const std::vector<float>& base_values,
    const std::vector<ThresholdType>& base_values_as_tensor,
    int64_t n_targets_or_classes,
    const std::vector<int64_t>& nodes_falsenodeids,
    const std::vector<int64_t>& nodes_featureids,
    const std::vector<float>& nodes_hitrates,
    const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
    const std::vector<int64_t>& nodes_missing_value_tracks_true,
    const std::vector<std::string>& nodes_modes,
    const std::vector<int64_t>& nodes_nodeids,
    const std::vector<int64_t>& nodes_treeids,
    const std::vector<int64_t>& nodes_truenodeids,
    const std::vector<float>& nodes_values,
    const std::vector<ThresholdType>& nodes_values_as_tensor,
    const std::string& post_transform,
    const std::vector<int64_t>& target_class_ids,
    const std::vector<int64_t>& target_class_nodeids,
    const std::vector<int64_t>& target_class_treeids,
    const std::vector<float>& target_class_weights,
    const std::vector<ThresholdType>& target_class_weights_as_tensor) {
  parallel_tree_ = parallel_tree;
  parallel_tree_N_ = parallel_tree_N;
  parallel_N_ = parallel_N;

  ORT_ENFORCE(n_targets_or_classes > 0);
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_featureids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_modes.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_nodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_treeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_truenodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_values.size() ||
              nodes_falsenodeids.size() == nodes_values_as_tensor.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_nodeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  ORT_ENFORCE(base_values.empty() || base_values_as_tensor.empty());
  ORT_ENFORCE(nodes_hitrates.empty() || nodes_hitrates_as_tensor.empty());
  ORT_ENFORCE(nodes_values.empty() || nodes_values_as_tensor.empty());
  ORT_ENFORCE(target_class_weights.empty() || target_class_weights_as_tensor.empty());

  aggregate_function_ = MakeAggregateFunction(aggregate_function);
  post_transform_ = MakeTransform(post_transform);
  if (!base_values_as_tensor.empty()) {
    ORT_ENFORCE(base_values.empty());
    base_values_ = base_values_as_tensor;
  } else {
    base_values_.reserve(base_values.size());
    for (size_t i = 0, limit = base_values.size(); i < limit; ++i) {
      base_values_.push_back(static_cast<ThresholdType>(base_values[i]));
    }
  }
  n_targets_or_classes_ = n_targets_or_classes;
  max_tree_depth_ = 1000;
  ORT_ENFORCE(nodes_modes.size() < std::numeric_limits<uint32_t>::max());

  // Additional members
  size_t limit;
  uint32_t i;
  InlinedVector<NODE_MODE> cmodes;
  cmodes.reserve(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (i = 0, limit = nodes_modes.size(); i < limit; ++i) {
    cmodes.push_back(MakeTreeNodeMode(nodes_modes[i]));
    if (cmodes[i] == NODE_MODE::LEAF) continue;
    if (fpos == -1) {
      fpos = static_cast<int>(i);
      continue;
    }
    if (cmodes[i] != cmodes[fpos]) same_mode_ = false;
  }

  n_nodes_ = nodes_treeids.size();
  limit = static_cast<size_t>(n_nodes_);
  InlinedVector<TreeNodeElementId> node_tree_ids;
  node_tree_ids.reserve(limit);
  nodes_.clear();
  nodes_.reserve(limit);
  tree_answer.resize(limit);
  leftLeaves.reserve(limit);
  myLeaves.reserve(limit);
  roots_.clear();
  std::unordered_map<TreeNodeElementId, size_t, TreeNodeElementId::hash_fn> node_tree_ids_map;
  node_tree_ids_map.reserve(limit);

  InlinedVector<size_t> truenode_ids, falsenode_ids;
  truenode_ids.reserve(limit);
  falsenode_ids.reserve(limit);
  max_feature_id_ = 0;

  // Build node_tree_ids and node_tree_ids_map and truenode_ids and falsenode_ids
  for (i = 0; i < limit; ++i) {
    TreeNodeElementId node_tree_id{static_cast<int>(nodes_treeids[i]), static_cast<int>(nodes_nodeids[i])};
    auto p = node_tree_ids_map.insert(std::pair<TreeNodeElementId, size_t>(node_tree_id, i));
    if (!p.second) {
      ORT_THROW("Node ", node_tree_id.node_id, " in tree ", node_tree_id.tree_id, " is already there.");
    }
    node_tree_ids.emplace_back(node_tree_id);
  }

  TreeNodeElementId coor;
  for (i = 0; i < limit; ++i) {
    if (cmodes[i] == NODE_MODE::LEAF) {
      truenode_ids.push_back(0);
      falsenode_ids.push_back(0);
    } else {
      TreeNodeElementId& node_tree_id = node_tree_ids[i];
      coor.tree_id = node_tree_id.tree_id;
      coor.node_id = static_cast<int>(nodes_truenodeids[i]);
      ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));

      auto found = node_tree_ids_map.find(coor);
      if (found == node_tree_ids_map.end()) {
        ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (truenode).");
      }
      if (found->second == truenode_ids.size()) {
        ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (truenode).");
      }
      truenode_ids.emplace_back(found->second);

      coor.node_id = static_cast<int>(nodes_falsenodeids[i]);
      ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));
      found = node_tree_ids_map.find(coor);
      if (found == node_tree_ids_map.end()) {
        ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (falsenode).");
      }
      if (found->second == falsenode_ids.size()) {
        ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (falsenode).");
      }
      falsenode_ids.emplace_back(found->second);
      // We could also check that truenode_ids[truenode_ids.size() - 1] != falsenode_ids[falsenode_ids.size() - 1]).
      // It is valid but no training algorithm would produce a tree where left and right nodes are the same.
    }
  }

  std::vector<size_t> used_node(nodes_treeids.size(), 0);
  leaves_offset.push_back(0);
  for (i = 0; i < n_nodes_; ++i) {
    if (used_node[i] == false) {
      LeftLeaves(i, cmodes, truenode_ids, falsenode_ids, used_node, 0);
      leaves_offset.push_back(leaves_offset.back() + myLeaves[i]);
    }
  }


  // Let's construct nodes_ such that the false branch is always the next element in nodes_.
  // updated_mapping will translates the old position of each node to the new node position in nodes_.
  std::vector<size_t> updated_mapping(nodes_treeids.size(), 0);
  int64_t previous_tree_id = -1;
  for (i = 0; i < n_nodes_; ++i) {
    if (previous_tree_id == -1 || (previous_tree_id != node_tree_ids[i].tree_id)) {
      // New tree.
      int64_t tree_id = node_tree_ids[i].tree_id;
      size_t root_position =
          AddNodes(i, cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor, nodes_values,
                   nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids);
      roots_.push_back(&nodes_[root_position]);
      previous_tree_id = tree_id;
    }
  }

  n_trees_ = roots_.size();
  if (((int64_t)nodes_.size()) != n_nodes_) {
    ORT_THROW("Number of nodes in nodes_ (", nodes_.size(), ") is different from n_nodes (", n_nodes_, ").");
  }

  // Sort targets
  InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices;
  indices.reserve(target_class_nodeids.size());
  for (i = 0, limit = target_class_nodeids.size(); i < limit; i++) {
    indices.emplace_back(
        std::pair<TreeNodeElementId, uint32_t>(TreeNodeElementId{target_class_treeids[i], target_class_nodeids[i]}, i));
  }

  std::sort(indices.begin(), indices.end());

  TreeNodeElementId ind;
  SparseValue<ThresholdType> w;
  size_t indi;
  for (indi = 0, limit = target_class_nodeids.size(); indi < limit; ++indi) {
    ind = indices[indi].first;
    i = indices[indi].second;
    auto found = node_tree_ids_map.find(ind);
    if (found == node_tree_ids_map.end()) {
      ORT_THROW("Unable to find node ", ind.tree_id, "-", ind.node_id, " (weights).");
    }

    TreeNodeElement<ThresholdType>& leaf = nodes_[updated_mapping[found->second]];
    if (leaf.is_not_leaf()) {
      // An exception should be raised in that case. But this case may happen in
      // models converted with an old version of onnxmltools. These weights are ignored.
      // ORT_THROW("Node ", ind.tree_id, "-", ind.node_id, " is not a leaf.");
      continue;
    }
    w.i = target_class_ids[i];
    w.value = target_class_weights_as_tensor.empty() ? static_cast<ThresholdType>(target_class_weights[i])
                                                     : target_class_weights_as_tensor[i];
    if (leaf.truenode_or_weight.weight_data.n_weights == 0) {
      leaf.truenode_or_weight.weight_data.weight = static_cast<int32_t>(weights_.size());
      leaf.value_or_unique_weight = w.value;
    }
    ++leaf.truenode_or_weight.weight_data.n_weights;
    weights_.push_back(w);
  }

  for (auto& leave : leaves_) {
    leaves_weight.push_back(weights_[leave.first].value);
  }

  has_missing_tracks_ = false;
  for (auto itm = nodes_missing_value_tracks_true.begin(); itm != nodes_missing_value_tracks_true.end(); ++itm) {
    if (*itm) {
      has_missing_tracks_ = true;
      break;
    }
  }

  sort(split_nodes_for_sorting.begin(), split_nodes_for_sorting.end(), [](const SplitNode& left, const SplitNode& right) {
    if (left.feature == right.feature) {
      if (left.threshold == right.threshold) {
        return left.tree_id < right.tree_id;
      }
      return left.threshold > right.threshold;
    }
    return left.feature < right.feature;
  });


  size_t deleted = 0;
  SplitNode* lastNode = &split_nodes_for_sorting[0];
  for (size_t i=1; i<split_nodes_for_sorting.size(); i++) {
    auto currNode = split_nodes_for_sorting[i];
    if (lastNode->feature == currNode.feature && lastNode->tree_id == currNode.tree_id && lastNode->threshold == currNode.threshold) {
      lastNode->bitvector &= currNode.bitvector;
      deleted++;
    }
    else {
      split_nodes_for_sorting[i - deleted] = currNode;
      lastNode = &split_nodes_for_sorting[i - deleted];
    }
  }
  split_nodes_for_sorting.resize(split_nodes_for_sorting.size() - deleted);

  std::vector<bool> filled_offset(max_feature_id_ + 1, false);
  offsets.resize(max_feature_id_ + 1, 0);

  for (auto& curr : split_nodes_for_sorting) {
    if (!filled_offset[curr.feature]) {
      offsets[curr.feature] = split_nodes_.size();
      filled_offset[curr.feature] = true;
    }

    split_nodes_.push_back({curr.tree_id, curr.threshold, curr.bitvector});
  }
  offsets.push_back(split_nodes_.size());

  for (int i=1; i<max_feature_id_; i++) {
    if (!filled_offset[i]) {
      offsets[i] = offsets[i - 1];
    }
  }

  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
void TreeEnsembleCommon<InputType, ThresholdType, OutputType>::LeftLeaves(
    const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
    const InlinedVector<size_t>& falsenode_ids, std::vector<size_t>& used_node, size_t left) {
  used_node[i] = true;

  if (cmodes[i] == 1) {
    myLeaves[i] = 1;
    return;
  }

  LeftLeaves(falsenode_ids[i], cmodes, truenode_ids, falsenode_ids, used_node, left);
  LeftLeaves(truenode_ids[i], cmodes, truenode_ids, falsenode_ids, used_node, left + myLeaves[falsenode_ids[i]]);

  leftLeaves[i] = left;
  myLeaves[i] = myLeaves[falsenode_ids[i]] + myLeaves[truenode_ids[i]];
}

template <typename InputType, typename ThresholdType, typename OutputType>
size_t TreeEnsembleCommon<InputType, ThresholdType, OutputType>::AddNode(
    const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
    const InlinedVector<size_t>& falsenode_ids, const std::vector<int64_t>& nodes_featureids,
    const std::vector<ThresholdType>& nodes_values_as_tensor, const std::vector<float>& node_values,
    const std::vector<int64_t>& nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping, int64_t tree_id,
    const InlinedVector<TreeNodeElementId>& node_tree_ids) {
  // Validate this index maps to the same tree_id as the one we should be building.
  if (node_tree_ids[i].tree_id != tree_id) {
    ORT_THROW("Tree id mismatch. Expected ", tree_id, " but got ", node_tree_ids[i].tree_id, " at position ", i);
  }

  if (updated_mapping[i] != 0) {
    // In theory we should not accept any cycles, however in practice LGBM conversion implements set membership via a
    // series of "Equals" nodes, with the true branches directed at the same child node (a cycle).
    // We may instead seek to formalize set membership in the future.
    return updated_mapping[i];
  }

  size_t node_pos = nodes_.size();
  updated_mapping[i] = node_pos;

  TreeNodeElement<ThresholdType> node;
  node.flags = static_cast<uint8_t>(cmodes[i]);
  node.node_pos = node_pos;
  node.feature_id = static_cast<int>(nodes_featureids[i]);
  if (node.feature_id > max_feature_id_) {
    max_feature_id_ = node.feature_id;
  }

  node.value_or_unique_weight =
      nodes_values_as_tensor.empty() ? static_cast<ThresholdType>(node_values[i]) : nodes_values_as_tensor[i];

  if (i < static_cast<size_t>(nodes_missing_value_tracks_true.size()) && nodes_missing_value_tracks_true[i] == 1) {
    node.flags |= static_cast<uint8_t>(MissingTrack::kTrue);
  }

  nodes_.push_back(std::move(node));
  if (!nodes_[node_pos].is_not_leaf()) {
    nodes_[node_pos].truenode_or_weight.weight_data.weight = 0;
    nodes_[node_pos].truenode_or_weight.weight_data.n_weights = 0;
  }

  uint64_t bitvector = 0;
  if (nodes_[node_pos].is_not_leaf()) {
    uint64_t temp = 1;

    for (size_t j=0; j<64; j++, temp *= 2) {
      if (j < leftLeaves[i] || j >= leftLeaves[i] + myLeaves[falsenode_ids[i]]) {
        bitvector += temp;
      }
    }

    split_nodes_for_sorting.push_back({node.feature_id, tree_id, node.value_or_unique_weight, bitvector});
  }
  else {
    leaves_.push_back({
      nodes_[node_pos].truenode_or_weight.weight_data.weight,
      nodes_[node_pos].truenode_or_weight.weight_data.n_weights
    });
  }

  return node_pos;
}

template <typename InputType, typename ThresholdType, typename OutputType>
size_t TreeEnsembleCommon<InputType, ThresholdType, OutputType>::AddNodes(
    const size_t i, const InlinedVector<NODE_MODE>& cmodes, const InlinedVector<size_t>& truenode_ids,
    const InlinedVector<size_t>& falsenode_ids, const std::vector<int64_t>& nodes_featureids,
    const std::vector<ThresholdType>& nodes_values_as_tensor, const std::vector<float>& node_values,
    const std::vector<int64_t>& nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping, int64_t tree_id,
    const InlinedVector<TreeNodeElementId>& node_tree_ids) {
  size_t node_pos = AddNode(i, cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor,
            node_values, nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids);

  if (nodes_[node_pos].is_not_leaf()) {
    size_t false_branch =
        AddNodes(falsenode_ids[i], cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor,
                 node_values, nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids);
    if (false_branch != node_pos + 1) {
      ORT_THROW("False node must always be the next node, but it isn't at index ", node_pos, " with flags ",
                static_cast<int>(nodes_[node_pos].flags));
    }
    nodes_[node_pos].falsenode_or_weight.ptr = &nodes_[false_branch];
    size_t true_branch =
        AddNodes(truenode_ids[i], cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor,
                 node_values, nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids);
    // We don't need to store the false branch pointer since we know it is always in the immediate next entry in nodes_.
    // nodes_[node_pos].falsenode_inc_or_n_weights.ptr = &nodes_[false_branch];
    nodes_[node_pos].truenode_or_weight.ptr = &nodes_[true_branch];
  }
  return node_pos;
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                         const Tensor* X,
                                                                         Tensor* Y,
                                                                         Tensor* label) const {
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorAverage<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::SUM:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorSum<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MIN:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMin<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MAX:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMax<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    default:
      ORT_THROW("Unknown aggregation function in TreeEnsemble.");
  }
}

template <typename InputType, typename ThresholdType, typename OutputType>
template <typename AGG>
void TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ComputeAgg(concurrency::ThreadPool* ttp,
                                                                          const Tensor* X, Tensor* Z,
                                                                          Tensor* label, const AGG& agg) const {
  if (X->Shape().NumDimensions() > 2) {
    ORT_THROW("TreeEnsemble only works on 1D, 2D tensors.");
  }
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  OutputType* z_data = Z->MutableData<OutputType>();
  const InputType* x_data = X->Data<InputType>();

  // auto max_num_threads = concurrency::ThreadPool::DegreeOfParallelism(ttp);
  // int64_t C = X->Shape().NumDimensions() == 2 ? X->Shape()[1] : 1;

  // int64_t* label_data = label == nullptr ? nullptr : label->MutableData<int64_t>();
  // ScoreValue<ThresholdType> score;
  // for (int64_t i = 0; i < N; i++) {
  //   score = {0, 0};

  //   for (int64_t j = 0; j < n_trees_; ++j) {
  //     agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[onnxruntime::narrow<size_t>(j)], x_data + i * stride));
  //   }

  //   agg.FinalizeScores1(
  //       z_data + i,
  //       score,
  //       label_data == nullptr ? nullptr : (label_data + i)
  //   );
  // }

  for (int64_t i = 0; i < N; i++) {
    *(z_data + i) = QuickScorer(x_data + i * stride);
  }
}  // namespace detail

#define TREE_FIND_VALUE(CMP)                                                                           \
  if (has_missing_tracks_) {                                                                           \
    while (root->is_not_leaf()) {                                                                      \
      val = x_data[root->feature_id];                                                                  \
      root = (val CMP root->value_or_unique_weight || (root->is_missing_track_true() && _isnan_(val))) \
                 ? root->truenode_or_weight.ptr                                                        \
                 : root->falsenode_or_weight.ptr;                                                                           \
    }                                                                                                  \
  } else {                                                                                             \
    while (root->is_not_leaf()) {                                                                      \
      val = x_data[root->feature_id];                                                                  \
      root = val CMP root->value_or_unique_weight ? root->truenode_or_weight.ptr : root->falsenode_or_weight.ptr;           \
    }                                                                                                  \
  }

inline bool _isnan_(float x) { return std::isnan(x); }
inline bool _isnan_(double x) { return std::isnan(x); }
inline bool _isnan_(int64_t) { return false; }
inline bool _isnan_(int32_t) { return false; }

template <typename InputType, typename ThresholdType, typename OutputType>
TreeNodeElement<ThresholdType>*
TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ProcessTreeNodeLeave(
    TreeNodeElement<ThresholdType>* root, const InputType* x_data) const {
  InputType val;
  if (same_mode_) {
    switch (root->mode()) {
      case NODE_MODE::BRANCH_LEQ:
        if (has_missing_tracks_) {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = (val <= root->value_or_unique_weight || (root->is_missing_track_true() && _isnan_(val)))
                       ? root->truenode_or_weight.ptr
                       : root->falsenode_or_weight.ptr;
          }
        } else {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = val <= root->value_or_unique_weight ? root->truenode_or_weight.ptr : root->falsenode_or_weight.ptr;
          }
        }
        break;
      case NODE_MODE::BRANCH_LT:
        TREE_FIND_VALUE(<)
        break;
      case NODE_MODE::BRANCH_GTE:
        TREE_FIND_VALUE(>=)
        break;
      case NODE_MODE::BRANCH_GT:
        TREE_FIND_VALUE(>)
        break;
      case NODE_MODE::BRANCH_EQ:
        TREE_FIND_VALUE(==)
        break;
      case NODE_MODE::BRANCH_NEQ:
        TREE_FIND_VALUE(!=)
        break;
      case NODE_MODE::LEAF:
        break;
    }
  } else {  // Different rules to compare to node thresholds.
    ThresholdType threshold;
    while (1) {
      val = x_data[root->feature_id];
      threshold = root->value_or_unique_weight;
      switch (root->mode()) {
        case NODE_MODE::BRANCH_LEQ:
          root = val <= threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::BRANCH_LT:
          root = val < threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                    : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::BRANCH_GTE:
          root = val >= threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::BRANCH_GT:
          root = val > threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                    : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::BRANCH_EQ:
          root = val == threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::BRANCH_NEQ:
          root = val != threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root->falsenode_or_weight.ptr;
          break;
        case NODE_MODE::LEAF:
          return root;
      }
    }
  }
  return root;
}

template <typename InputType, typename ThresholdType, typename OutputType>
OutputType
TreeEnsembleCommon<InputType, ThresholdType, OutputType>::QuickScorer(const InputType* x_data) const {
  for (size_t i=0; i<(size_t)n_trees_; i++) {
    tree_answer[i] = -1;
  }

  for (size_t i = 0; i <= (size_t)max_feature_id_; i ++) {
    size_t it = offsets[i];
    size_t end = offsets[i + 1];
    while (x_data[i] <= split_nodes_[it].threshold && it < end) {
      tree_answer[split_nodes_[it].tree_id] &= split_nodes_[it].bitvector;
      it++;
    }
  }

  ThresholdType score = 0;
  for (size_t i=0; i<(size_t)n_trees_; i++) {
    auto j = __builtin_ctzl(tree_answer[i]);
    score += leaves_weight[leaves_offset[i] + j];
  }
  score /= n_trees_;

  return score;
}

// TI: input type
// TH: threshold type, double if T==double, float otherwise
// TO: output type
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommonClassifier : public TreeEnsembleCommon<InputType, ThresholdType, OutputType> {
 private:
  bool weights_are_all_positive_;
  bool binary_case_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<int64_t> class_labels_;

 public:
  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Z, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_tree_N,
              int parallel_N,
              const std::string& aggregate_function,
              const std::vector<float>& base_values,
              const std::vector<ThresholdType>& base_values_as_tensor,
              const std::vector<int64_t>& nodes_falsenodeids,
              const std::vector<int64_t>& nodes_featureids,
              const std::vector<float>& nodes_hitrates,
              const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
              const std::vector<int64_t>& nodes_missing_value_tracks_true,
              const std::vector<std::string>& nodes_modes,
              const std::vector<int64_t>& nodes_nodeids,
              const std::vector<int64_t>& nodes_treeids,
              const std::vector<int64_t>& nodes_truenodeids,
              const std::vector<float>& nodes_values,
              const std::vector<ThresholdType>& nodes_values_as_tensor,
              const std::string& post_transform,
              const std::vector<int64_t>& class_ids,
              const std::vector<int64_t>& class_nodeids,
              const std::vector<int64_t>& class_treeids,
              const std::vector<float>& class_weights,
              const std::vector<ThresholdType>& class_weights_as_tensor,
              const std::vector<std::string>& classlabels_strings,
              const std::vector<int64_t>& classlabels_int64s);
};

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  std::vector<ThresholdType> base_values_as_tensor, nodes_hitrates_as_tensor,
      nodes_values_as_tensor, class_weights_as_tensor;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "class_weights_as_tensor", class_weights_as_tensor));
#endif

  return Init(
      80,
      128,
      50,
      info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
      info.GetAttrsOrDefault<float>("base_values"),
      base_values_as_tensor,
      info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_featureids"),
      info.GetAttrsOrDefault<float>("nodes_hitrates"),
      nodes_hitrates_as_tensor,
      info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"),
      info.GetAttrsOrDefault<std::string>("nodes_modes"),
      info.GetAttrsOrDefault<int64_t>("nodes_nodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_treeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"),
      info.GetAttrsOrDefault<float>("nodes_values"),
      nodes_values_as_tensor,
      info.GetAttrOrDefault<std::string>("post_transform", "NONE"),
      info.GetAttrsOrDefault<int64_t>("class_ids"),
      info.GetAttrsOrDefault<int64_t>("class_nodeids"),
      info.GetAttrsOrDefault<int64_t>("class_treeids"),
      info.GetAttrsOrDefault<float>("class_weights"),
      class_weights_as_tensor,
      info.GetAttrsOrDefault<std::string>("classlabels_strings"),
      info.GetAttrsOrDefault<int64_t>("classlabels_int64s"));
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
    int parallel_N,
    const std::string& aggregate_function,
    const std::vector<float>& base_values,
    const std::vector<ThresholdType>& base_values_as_tensor,
    const std::vector<int64_t>& nodes_falsenodeids,
    const std::vector<int64_t>& nodes_featureids,
    const std::vector<float>& nodes_hitrates,
    const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
    const std::vector<int64_t>& nodes_missing_value_tracks_true,
    const std::vector<std::string>& nodes_modes,
    const std::vector<int64_t>& nodes_nodeids,
    const std::vector<int64_t>& nodes_treeids,
    const std::vector<int64_t>& nodes_truenodeids,
    const std::vector<float>& nodes_values,
    const std::vector<ThresholdType>& nodes_values_as_tensor,
    const std::string& post_transform,
    const std::vector<int64_t>& class_ids,
    const std::vector<int64_t>& class_nodeids,
    const std::vector<int64_t>& class_treeids,
    const std::vector<float>& class_weights,
    const std::vector<ThresholdType>& class_weights_as_tensor,
    const std::vector<std::string>& classlabels_strings,
    const std::vector<int64_t>& classlabels_int64s) {
  auto status = TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(
      parallel_tree,
      parallel_tree_N,
      parallel_N,
      aggregate_function,
      base_values,
      base_values_as_tensor,
      classlabels_strings.empty() ? classlabels_int64s.size()
                                  : classlabels_strings.size(),
      nodes_falsenodeids,
      nodes_featureids,
      nodes_hitrates,
      nodes_hitrates_as_tensor,
      nodes_missing_value_tracks_true,
      nodes_modes,
      nodes_nodeids,
      nodes_treeids,
      nodes_truenodeids,
      nodes_values,
      nodes_values_as_tensor,
      post_transform,
      class_ids,
      class_nodeids,
      class_treeids,
      class_weights,
      class_weights_as_tensor);
  ORT_RETURN_IF_ERROR(status);

  classlabels_strings_ = classlabels_strings;
  classlabels_int64s_ = classlabels_int64s;

  InlinedHashSet<int64_t> weights_classes;
  weights_classes.reserve(class_ids.size());
  weights_are_all_positive_ = true;
  for (size_t i = 0, end = class_ids.size(); i < end; ++i) {
    weights_classes.insert(class_ids[i]);
    if (weights_are_all_positive_ && (!class_weights.empty() ? class_weights[i] : class_weights_as_tensor[i]) < 0)
      weights_are_all_positive_ = false;
  }
  binary_case_ = this->n_targets_or_classes_ == 2 && weights_classes.size() == 1;
  if (!classlabels_strings_.empty()) {
    class_labels_.reserve(classlabels_strings_.size());
    for (size_t i = 0, end = classlabels_strings_.size(); i < end; ++i)
      class_labels_.push_back(i);
  }
  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                                   const Tensor* X,
                                                                                   Tensor* Z,
                                                                                   Tensor* label) const {
  if (classlabels_strings_.empty()) {
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, label,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            classlabels_int64s_, binary_case_,
            weights_are_all_positive_));
  } else {
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    AllocatorPtr alloc;
    ORT_THROW_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    Tensor label_int64(DataTypeImpl::GetType<int64_t>(), TensorShape({N}), std::move(alloc));
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, &label_int64,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            class_labels_, binary_case_,
            weights_are_all_positive_));
    const int64_t* plabel = label_int64.Data<int64_t>();
    std::string* labels = label->MutableData<std::string>();
    for (size_t i = 0; i < (size_t)N; ++i)
      labels[i] = classlabels_strings_[onnxruntime::narrow<size_t>(plabel[i])];
  }
  return Status::OK();
}

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
