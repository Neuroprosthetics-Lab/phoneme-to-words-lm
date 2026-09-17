#pragma once

#include "flashlight/lib/text/Defines.h"
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

/**
 * Add per-word bonuses in the inner LM's score units (log10 for KenLM).
 * States, sentence boundaries and caches belong to the inner LM. No trie
 * lookahead is changed here. Replace bonuses only between utterances; this
 * object, like its inner LM, is not intended for concurrent mutation/decoding.
 */
class FL_TEXT_API HotwordLM : public LM {
 public:
  using Bonuses = std::unordered_map<int, double>;

  explicit HotwordLM(LMPtr inner, Bonuses bonuses = {});

  LMStatePtr start(bool startWithNothing) override;
  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state, int usrTokenIdx) override;
  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;
  void updateCache(std::vector<LMStatePtr> states) override;

  const Bonuses& getHotwordBonus() const;
  // Validate a replacement before publishing it. Invalid updates leave the
  // previous table intact. Doubles preserve Python BiasingLM's rounding.
  void setHotwordBonus(Bonuses bonuses);

 private:
  LMPtr inner_;
  Bonuses bonuses_;
};

using HotwordLMPtr = std::shared_ptr<HotwordLM>;

} // namespace text
} // namespace lib
} // namespace fl
