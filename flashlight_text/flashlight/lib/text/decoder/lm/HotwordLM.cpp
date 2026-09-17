#include "flashlight/lib/text/decoder/lm/HotwordLM.h"

#include <cmath>
#include <stdexcept>

namespace fl {
namespace lib {
namespace text {

HotwordLM::HotwordLM(LMPtr inner, Bonuses bonuses) : inner_(std::move(inner)) {
  if (!inner_) {
    throw std::invalid_argument("HotwordLM requires an inner LM");
  }
  setHotwordBonus(std::move(bonuses));
}

LMStatePtr HotwordLM::start(bool startWithNothing) {
  return inner_->start(startWithNothing);
}

std::pair<LMStatePtr, float> HotwordLM::score(
    const LMStatePtr& state, int usrTokenIdx) {
  auto result = inner_->score(state, usrTokenIdx);
  const auto bonus = bonuses_.find(usrTokenIdx);
  if (bonus != bonuses_.end()) {
    // Python receives the inner float as a double, adds a double bonus, then
    // PyLM casts back to float. Do not round the bonus before the addition.
    result.second = static_cast<float>(
        static_cast<double>(result.second) + bonus->second);
  }
  return result;
}

std::pair<LMStatePtr, float> HotwordLM::finish(const LMStatePtr& state) {
  return inner_->finish(state);
}

void HotwordLM::updateCache(std::vector<LMStatePtr> states) {
  inner_->updateCache(std::move(states));
}

const HotwordLM::Bonuses& HotwordLM::getHotwordBonus() const {
  return bonuses_;
}

void HotwordLM::setHotwordBonus(Bonuses bonuses) {
  for (const auto& entry : bonuses) {
    if (entry.first < 0 || !std::isfinite(entry.second)) {
      throw std::invalid_argument(
          "HotwordLM requires nonnegative word indices and finite bonuses");
    }
  }
  bonuses_.swap(bonuses);
}

} // namespace text
} // namespace lib
} // namespace fl
