#ifndef RecoAlgos_SingleElementCollectionSelectorPlusEvent_h
#define RecoAlgos_SingleElementCollectionSelectorPlusEvent_h
/** \class SingleElementCollectionSelectorPlusEvent
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleElementCollectionSelectorPlusEvent.h,v 1.1 2009/03/03 13:07:28 llista Exp $
 *
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

template <typename InputCollection,
          typename Selector,
          typename OutputCollection = typename helper::SelectedOutputCollectionTrait<InputCollection>::type,
          typename StoreContainer = typename helper::StoreContainerTrait<OutputCollection>::type,
          typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
struct SingleElementCollectionSelectorPlusEvent {
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionSelectorPlusEvent(const edm::ParameterSet &cfg, edm::ConsumesCollector &&iC)
      : select_(reco::modules::make<Selector>(cfg, iC)) {}
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection> &c, const edm::Event &ev, const edm::EventSetup &) {
    selected_.clear();
    for (size_t idx = 0; idx < c->size(); ++idx) {
      if (select_(edm::Ref<InputCollection>(c, idx), ev))  //(*c)[idx]
        addRef_(selected_, c, idx);
    }
  }

  static void fillPSetDescription(edm::ParameterSetDescription &desc) { Selector::fillPSetDescription(desc); };

private:
  StoreContainer selected_;
  Selector select_;
  RefAdder addRef_;
};

#endif
