#include <cstdlib>
#include <cassert>
#include <iostream>

using std::cerr;


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
template <size_t a,size_t b> struct Mul {};
template <size_t x,size_t y,size_t z> struct Vec3 {};
template <size_t row_0,size_t row_1,size_t row_2> struct Mat33 {};
template <size_t index> struct XValue {};
template <size_t index> struct YValue {};
template <size_t index> struct ZValue {};

template <size_t index,typename Expr> struct Node {};
template <size_t mat_index,size_t row,size_t col> struct Elem {};

struct Empty {};
struct None {};

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
template <typename Tag,typename T>
struct Let {
  const T value;
};
}


namespace {
template <typename First,typename Tag,typename T> struct LetList {
  First first;
  const T value;
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
template <template<size_t...> typename Op,size_t... indices,typename Map>
auto mapExpr(Op<indices...>, Map)
{
  return Op<mapped_index<indices,Map>...>{};
}
}


namespace {
template <size_t mat_index,size_t row,size_t col,typename Map>
auto mapExpr(Elem<mat_index,row,col>, Map)
{
  return Elem<mapped_index<mat_index,Map>,row,col>{};
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
template <typename Expr>
auto findNodeIndex(Expr,List<>)
{
  return None{};
}
}


namespace {
template <
  typename Expr,
  size_t index2,
  typename Expr2,
  typename... Nodes
>
auto
  findNodeIndex(
    Expr,
    List<Node<index2, Expr2>, Nodes...>
  )
{
  return findNodeIndex(Expr{},List<Nodes...>{});
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
template <typename MergedNodes,typename NewExpr>
auto mergedGraph(MergedNodes,NewExpr)
{
  auto new_nodes = addNode(MergedNodes{},NewExpr{});
  return Graph<listSize<MergedNodes>,decltype(new_nodes)>{};
}
}


namespace {
template <
  template <size_t,size_t> typename Op,
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto binary(Graph<index_a,NodesA>,Graph<index_b,NodesB>)
{
  constexpr size_t new_index_a = index_a;
  using MergeResult = decltype(merge(NodesA{},NodesB{}));
  using MergedNodes = typename MergeResult::Nodes;
  using MapB = typename MergeResult::MapB;
  constexpr size_t new_index_b = mapped_index<index_b,MapB>;
  return mergedGraph(MergedNodes{},Op<new_index_a,new_index_b>{});
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto operator+(Graph<index_a,NodesA> graph_a,Graph<index_b,NodesB> graph_b)
{
  return binary<Add>(graph_a,graph_b);
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto operator*(Graph<index_a,NodesA> graph_a,Graph<index_b,NodesB> graph_b)
{
  return binary<Mul>(graph_a,graph_b);
}
}


namespace {
template <typename Tag,typename T>
auto let(Graph<0,List<Node<0,Var<Tag>>>>,const T& value)
{
  return Let<Tag,T>{value};
}
}


namespace {
template <typename First,typename Tag,typename T>
auto letList(First first,Let<Tag,T> last_let)
{
  return LetList<First,Tag,T>{first,last_let.value};
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
template <typename Tag,typename First,typename T>
auto getLet(Tagged<Tag>,const LetList<First, Tag,T>& lets)
{
  return lets.value;
}
}


namespace {
template <typename Tag,typename Tag2,typename First,typename T2>
auto getLet(Tagged<Tag>,const LetList<First, Tag2, T2>& lets)
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
struct Vec3f {
  const float x,y,z;

  bool operator==(const Vec3f &arg) const
  {
    return x==arg.x && y==arg.y && z==arg.z;
  }
};
}


static Vec3f vec3(float x,float y,float z)
{
  return Vec3f{x,y,z};
}


static float xValue(const Vec3f &v) { return v.x; }
static float yValue(const Vec3f &v) { return v.y; }
static float zValue(const Vec3f &v) { return v.z; }


namespace {
template <typename A,typename Values,typename Lets>
auto evalExpr(Var<A>, const Values &,const Lets &lets)
{
  return getLet(Tagged<A>{},lets);
}
}


namespace {
template <size_t a_index,size_t b_index>
auto evalOp(Add<a_index,b_index>,float a,float b)
{
  return a+b;
}
}


namespace {
template <size_t a_index,size_t b_index>
auto evalOp(Mul<a_index,b_index>,float a,float b)
{
  return a*b;
}
}


namespace {
template <size_t x_index,size_t y_index,size_t z_index>
auto evalOp(Vec3<x_index,y_index,z_index>,float x,float y,float z)
{
  return vec3(x,y,z);
}
}


namespace {

template <size_t i> auto evalOp(XValue<i>,const Vec3f &v) { return xValue(v); }
template <size_t i> auto evalOp(YValue<i>,const Vec3f &v) { return yValue(v); }
template <size_t i> auto evalOp(ZValue<i>,const Vec3f &v) { return zValue(v); }

}



namespace {
template <
  template <size_t...> typename Op,
  size_t... indices,
  typename Values,
  typename Lets
>
auto evalExpr(Op<indices...>, const Values &values,const Lets &)
{
  return evalOp(Op<indices...>{},getValue(Indexed<indices>{},values)...);
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


template <
  size_t x_index,size_t y_index,size_t z_index,
  typename XNodes,typename YNodes,typename ZNodes
>
static auto vec3(
  Graph<x_index,XNodes>,
  Graph<y_index,YNodes>,
  Graph<z_index,ZNodes>
)
{
  constexpr size_t new_x_index = x_index;
  using XYMergeResult = decltype(merge(XNodes{},YNodes{}));
  using XYNodes = typename XYMergeResult::Nodes;
  using MapY = typename XYMergeResult::MapB;
  constexpr size_t new_y_index = mapped_index<y_index,MapY>;
  using XYZMergeResult = decltype(merge(XYNodes{},ZNodes{}));
  using XYZNodes = typename XYZMergeResult::Nodes;
  using MapZ = typename XYZMergeResult::MapB;
  constexpr size_t new_z_index = mapped_index<z_index,MapZ>;

  return mergedGraph(XYZNodes{},Vec3<new_x_index,new_y_index,new_z_index>{});
}


namespace {
struct Mat33f {
  const float values[3][3];

  Mat33f(const float (&m)[3][3])
  : values{
      {m[0][0],m[0][1],m[0][2]},
      {m[1][0],m[1][1],m[1][2]},
      {m[2][0],m[2][1],m[2][2]},
    }
  {
  }

  bool operator==(const Mat33f &arg) const
  {
    for (int i=0; i!=3; ++i) {
      for (int j=0; j!=3; ++j) {
        if (values[i][j] != arg.values[i][j]) {
          return false;
        }
      }
    }

    return true;
  }
};
}


namespace {
template <
  size_t mat_index,
  size_t row,
  size_t col,
  typename Values,
  typename Lets
>
auto evalExpr(Elem<mat_index,row,col>, const Values &values,const Lets &)
{
  Mat33f mat_value = getValue(Indexed<mat_index>{},values);
  return mat_value.values[row][col];
}
}


namespace {
Mat33f mat33(Vec3f r1,Vec3f r2,Vec3f r3)
{
  float values[3][3] = {
    {r1.x,r1.y,r1.z},
    {r2.x,r2.y,r2.z},
    {r3.x,r3.y,r3.z},
  };

  return Mat33f{values};
}
}




template <
  size_t row0_index,size_t row1_index,size_t row2_index,
  typename Row0Nodes,typename Row1Nodes,typename Row2Nodes
>
static auto mat33(
  Graph<row0_index,Row0Nodes>,
  Graph<row1_index,Row1Nodes>,
  Graph<row2_index,Row2Nodes>
)
{
  constexpr size_t new_row0_index = row0_index;
  using Row01MergeResult = decltype(merge(Row0Nodes{},Row1Nodes{}));
  using Row01Nodes = typename Row01MergeResult::Nodes;
  using MapRow1 = typename Row01MergeResult::MapB;
  constexpr size_t new_row1_index = mapped_index<row1_index,MapRow1>;
  using Row012MergeResult = decltype(merge(Row01Nodes{},Row2Nodes{}));
  using Row012Nodes = typename Row012MergeResult::Nodes;
  using MapRow2 = typename Row012MergeResult::MapB;
  constexpr size_t new_row2_index = mapped_index<row2_index,MapRow2>;

  return
    mergedGraph(
      Row012Nodes{},
      Vec3<new_row0_index,new_row1_index,new_row2_index>{}
    );
}


template <size_t index,typename Nodes>
static auto xValue(Graph<index,Nodes>)
{
  return mergedGraph(Nodes{},XValue<index>{});
}


template <size_t index,typename Nodes>
static auto yValue(Graph<index,Nodes>)
{
  return mergedGraph(Nodes{},YValue<index>{});
}


template <size_t index,typename Nodes>
static auto zValue(Graph<index,Nodes>)
{
  return mergedGraph(Nodes{},ZValue<index>{});
}


template <typename A,typename B>
static auto dot(A a,B b)
{
  return
    xValue(a)*xValue(b) +
    yValue(a)*yValue(b) +
    zValue(a)*zValue(b);
}


static void testFindNodeIndex()
{
  struct A;
  struct B;

  using MaybeMergedIndex =
    decltype(
      findNodeIndex(
        Var<B>{},
        List<
          Node<0,Var<A>>,
          Node<1,Var<B>>
        >{}
      )
    );

  static_assert(MaybeMergedIndex::value == 1);
}


template <size_t row,size_t col,size_t mat_index,typename Nodes>
static auto elem(Graph<mat_index,Nodes>)
{
  return mergedGraph(Nodes{},Elem<mat_index,row,col>{});
}


template <size_t index,typename M>
static auto col(const M &m)
{
  auto x = elem<0,index>(m);
  auto y = elem<1,index>(m);
  auto z = elem<2,index>(m);
  return vec3(x,y,z);
}


int main()
{
  testFindNodeIndex();
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
  {
    auto a = var<struct A>();
    auto b = var<struct B>();
    auto c = a*b;
    assert(eval(c,let(a,3),let(b,4)) == 3*4);
  }
  {
    auto ax = var<struct A>();
    auto ay = var<struct B>();
    auto az = var<struct C>();
    auto v = vec3(ax,ay,az);
    auto result = eval(v,let(ax,1),let(ay,2),let(az,3));
    auto expected_result = vec3(1,2,3);
    assert(result == expected_result);
  }
  {
    auto ax = var<struct AX>();
    auto ay = var<struct AY>();
    auto az = var<struct AZ>();
    auto bx = var<struct BX>();
    auto by = var<struct BY>();
    auto bz = var<struct BZ>();
    auto a = vec3(ax,ay,az);
    auto b = vec3(bx,by,bz);
    auto c = dot(a,b);

    auto result =
      eval(c,let(ax,1),let(ay,2),let(az,3),let(bx,4),let(by,5),let(bz,6));

    auto expected_result = dot(vec3(1,2,3),vec3(4,5,6));
    assert(result == expected_result);
  }
  {
    auto a = var<struct A>();
    auto aT0 = col<0>(a);
    Vec3f row0 = vec3(1,2,3);
    Vec3f row1 = vec3(4,5,6);
    Vec3f row2 = vec3(7,8,9);
    auto result = eval(aT0,let(a,mat33(row0,row1,row2)));
    auto expected_result = vec3(xValue(row0),xValue(row1),xValue(row2));
    assert(result == expected_result);
  }
}
