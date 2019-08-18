#include <cstdlib>
#include <cassert>

namespace {

template <typename Tag> struct Var { };
template <typename Tag> struct Tagged {};

template <size_t index> struct Indexed
{
  static constexpr auto value = index;
};

template <size_t expr_index,typename Nodes> struct Graph {};

template <typename...> struct List {};
template <size_t key,size_t value> struct MapEntry {};
template <size_t a,size_t b> struct Add {};
template <size_t index,typename Expr> struct Node {};

struct Empty {};
struct None{};

}


namespace {
template <typename First,size_t index,typename T>
struct IndexedValueList {
  First first;
  T value;
};
}


namespace {
template <typename NodesArg,typename MapBArg>
struct MergeResult {
  using Nodes = NodesArg;
  using MapB = MapBArg;
};
}


namespace {
template <typename Tag>
struct Let {
  const float value;
};
}


namespace {
template <typename First,typename Tag> struct LetList {
  First first;
  const float value;
};
}



namespace {
template <
  size_t expr_index,size_t value_index,typename... Entries
>
Indexed<value_index>
  findNewIndex(
    Indexed<expr_index>,
    List<MapEntry<expr_index,value_index>,
    Entries...>
  )
{
  return {};
}
}


namespace {
template <typename Index,typename FirstEntry, typename... Entries>
auto findNewIndex(Index, List<FirstEntry, Entries...>)
{
  return findNewIndex(Index{}, List<Entries...>{});
}
}


template <size_t index,typename Map>
static constexpr size_t mapped_index =
  decltype(findNewIndex(Indexed<index>{},Map{}))::value;
  

namespace {
template <typename Tag>
auto var()
{
  using Expr = Var<Tag>;
  using Node0 = Node<0,Expr>;
  return Graph<0,List<Node0>>{};
}
}


namespace {
template <typename A,typename Map>
auto mapExpr(Var<A>,Map)
{
  return Var<A>{};
}
}


namespace {
template <size_t index1, size_t index2, typename Map>
auto mapExpr(Add<index1, index2>, Map)
{
  constexpr size_t new_index1 = mapped_index<index1,Map>;
  constexpr size_t new_index2 = mapped_index<index2,Map>;
  return Add<new_index1,new_index2>{};
}
}


namespace {
template <typename Expr,size_t index,typename... Nodes>
auto findNodeIndex(Expr,List<Node<index,Expr>,Nodes...>)
{
  return Indexed<index>{};
}
}


namespace {
template <
  typename Expr,
  size_t index2,
  typename Expr2,
  typename... Nodes
>
auto findNodeIndex(Expr, List<Node<index2, Expr2>,Nodes...>)
{
  return None{};
}
}


namespace {
template <typename NewMergedNodesArg,size_t new_index_arg>
struct UpdateMergedNodesResult {
  using NewMergedNodes = NewMergedNodesArg;
  static constexpr size_t new_index = new_index_arg;
};
}


namespace {
template <typename... Nodes,size_t new_index,typename Expr>
auto updateMergedNodes(List<Nodes...>,Indexed<new_index>,Expr)
{
  using NewMergedNodes = List<Nodes...>;
  return UpdateMergedNodesResult<NewMergedNodes,new_index>{};
}
}


namespace {
template <typename... Nodes,typename Expr>
auto updateMergedNodes(List<Nodes...>,None,Expr)
{
  static constexpr size_t new_index = sizeof...(Nodes);
  using NewMergedNodes = List<Nodes...,Node<new_index,Expr>>;
  return UpdateMergedNodesResult<NewMergedNodes,new_index>{};
}
}


namespace {
// If there are no more nodes to add, return what we've build.
template <typename NewMergedNodes,typename NewMapB>
auto buildMergedNodes(NewMergedNodes, List<>, NewMapB)
{
  return MergeResult<NewMergedNodes,NewMapB>{};
}
}


namespace {
// If we have nodes to add, add the first one and recurse.
template <
  typename MergedNodes,
  typename... BNodes,
  size_t index_b,
  typename ExprB,
  typename... MapBEntries
>
auto
  buildMergedNodes(
    MergedNodes,
    List<Node<index_b,ExprB>,BNodes...>,
    List<MapBEntries...>
  )
{
  using MappedExpr = decltype(mapExpr(ExprB{},List<MapBEntries...>{}));
  using MaybeMergedIndex = decltype(findNodeIndex(MappedExpr{},MergedNodes{}));

  using UpdateResult =
    decltype(updateMergedNodes(MergedNodes{},MaybeMergedIndex{},MappedExpr{}));

  using NewMergedNodes = typename UpdateResult::NewMergedNodes;
  constexpr auto new_index = UpdateResult::new_index;

  using NewMapB = List<MapBEntries...,MapEntry<index_b,new_index>>;
  return buildMergedNodes(NewMergedNodes{},List<BNodes...>{},NewMapB{});
}
}


namespace {
// Build the merged nodes by starting with the first list of nodes and
// adding the second list.
template <typename... NodesA,typename...NodesB>
auto merge(List<NodesA...>,List<NodesB...>)
{
  using MergedNodes = List<NodesA...>;
  using MapEntries = List<>;
  return buildMergedNodes(MergedNodes{},List<NodesB...>{},MapEntries{});
}
}


namespace {

template <typename... Nodes,typename Expr>
auto addNode(List<Nodes...>,Expr)
{
  return List<Nodes..., Node<sizeof...(Nodes),Expr>>{};
}

}


namespace {
template <typename L>
struct ListSize;

template <typename... Elements>
struct ListSize<List<Elements...>> {
  static constexpr size_t value = sizeof...(Elements);
};
}


namespace {
template <typename L>
constexpr size_t listSize = ListSize<L>::value;
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto operator+(Graph<index_a,NodesA>,Graph<index_b,NodesB>)
{
  auto merged = merge(NodesA{},NodesB{});
  using MergeResult = decltype(merged);
  using MergedNodes = typename MergeResult::Nodes;
  using MapB = typename MergeResult::MapB;
  using NewExprIndexB = decltype(findNewIndex(Indexed<index_b>{},MapB{}));

  auto new_nodes =
    addNode(
      MergedNodes{},
      Add<index_a,NewExprIndexB::value>{}
    );

  return Graph<listSize<MergedNodes>,decltype(new_nodes)>{};
}
}


namespace {
template <typename Tag>
auto let(Graph<0,List<Node<0,Var<Tag>>>>,float value)
{
  return Let<Tag>{value};
}
}


namespace {
template <typename First,typename Tag>
auto letList(First first,Let<Tag> last_let)
{
  return LetList<First,Tag>{first,last_let.value};
}
}


namespace {
template <typename Result>
auto buildLetList(Result result)
{
  return result;
}
}


namespace {
template <typename Result,typename FirstLet,typename... RestLets>
auto buildLetList(Result result,FirstLet first_let,RestLets... rest_lets)
{
  return buildLetList(letList(result,first_let),rest_lets...);
}
}


namespace {
template <typename T>
auto valueList(Empty,T value)
{
  return IndexedValueList<Empty,0,T>{Empty{},value};
}
}


namespace {
template <size_t index,typename First,typename T1,typename T>
auto valueList(IndexedValueList<First,index,T1> values,T value)
{
  return
    IndexedValueList<IndexedValueList<First,index,T1>,index+1,T>{
      values,
      value
    };
}
}


namespace {
template <typename Tag,typename First>
auto getLet(Tagged<Tag>,const LetList<First, Tag>& lets)
{
  return lets.value;
}
}


namespace {
template <typename Tag,typename Tag2,typename First>
auto getLet(Tagged<Tag>,const LetList<First, Tag2>& lets)
{
  return getLet(Tagged<Tag>{},lets.first);
}
}


namespace {
template <size_t index,typename First,typename T>
auto getValue(Indexed<index>,IndexedValueList<First,index,T> list)
{
  return list.value;
}
}


namespace {
template <size_t index,size_t index2,typename T, typename First>
auto getValue(Indexed<index>, const IndexedValueList<First, index2, T>& values)
{
  return getValue(Indexed<index>{}, values.first);
}
}


namespace {
template <typename A,typename Values,typename Lets>
auto evalExpr(Var<A>, const Values &,const Lets &lets)
{
  return getLet(Tagged<A>{},lets);
}
}


namespace {
template <size_t index1,size_t index2,typename Values,typename Lets>
auto evalExpr(Add<index1,index2>, const Values &values,const Lets &)
{
  auto value1 = getValue(Indexed<index1>{},values);
  auto value2 = getValue(Indexed<index2>{},values);
  return value1 + value2;
}
}


namespace {
template <typename Values,typename Lets>
auto evalNodes(Values values,List<>,const Lets &)
{
  return values;
}
}


namespace {
template <
  typename Values,
  typename Lets,
  size_t index,
  typename Expr,
  typename... Nodes
>
auto evalNodes(Values values,List<Node<index,Expr>,Nodes...>,Lets lets)
{
  auto value = evalExpr(Expr{}, values, lets);
  return evalNodes(valueList(values,value),List<Nodes...>{},lets);
}
}


namespace {
template <typename Nodes,size_t index,typename ...Lets>
auto
  eval(
    Graph<index, Nodes>,
    Lets... lets
  )
{
  auto values =
    evalNodes(
      Empty{},
      Nodes{},
      buildLetList(Empty{},lets...)
    );


  return getValue(Indexed<index>{},values);
}
}


int main()
{
  {
    auto a = var<struct A>();
    auto b = a+a;
    assert(eval(b,let(a,3)) == 6);
  }
  {
    auto a = var<struct A>();
    auto b = var<struct B>();
    auto c = a+b;
    assert(eval(c,let(a,3),let(b,4)) == 3+4);
  }
  {
    auto a = var<struct A>();
    auto b = var<struct B>();
    auto c = a+b+a;
    assert(eval(c,let(a,3),let(b,4)) == 3+4+3);
  }
  {
    auto a = var<struct A>();
    auto b = var<struct B>();
    auto c = (a+a)+(b+b);
    assert(eval(c,let(a,3),let(b,4)) == (3+3)+(4+4));
  }
}
