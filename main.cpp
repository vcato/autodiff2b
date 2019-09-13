#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>
#include <type_traits>

#define ADD_QR_DECOMP 0


using std::cerr;
using std::is_same_v;


namespace {

template <typename Map,size_t default_value>
struct MapWithDefault {};

}


namespace {

template <typename Tag> struct Var { };
template <typename ValueType> struct Const { };
template <typename Tag> struct Tagged {};

}


namespace {

template <size_t index> struct Indexed
{
  static constexpr auto value = index;
};

}


namespace {
template <size_t...> struct Indices {};
}


namespace {

template <typename Output,typename Nodes> struct Graph { };

template <typename Output,typename Nodes>
Nodes nodesOf(Graph<Output,Nodes>);

template <typename Output,typename Nodes>
Output outputOf(Graph<Output,Nodes>);

}


namespace {
template <size_t> struct ScalarIndices {};
}


namespace {
template <size_t index,typename Nodes>
struct Graph<ScalarIndices<index>,Nodes> {
  operator float() const;
};
}


namespace {

template <size_t expr_index,typename Nodes>
constexpr auto indexOf(Graph<ScalarIndices<expr_index>,Nodes>)
{
  return Indexed<expr_index>{};
}

}



namespace {

template <typename...> struct List {};


template <typename... Nodes>
static constexpr size_t sizeOf(List<Nodes...>)
{
  return sizeof...(Nodes);
}


}


namespace {

template <size_t key,size_t value> struct MapEntry {};

}


namespace {

struct Zero {
  static float value() { return 0; }
};


struct One {
  static float value() { return 1; }
};


struct Half {
  static float value() { return 0.5; }
};

}


namespace {

template <size_t a_index,size_t b_index>
struct Add {
  static float eval(float a,float b) { return a+b; }
};


template <size_t a_index,size_t b_index>
struct Sub
{
  template <typename A,typename B>
  static auto eval(const A& a,const B& b) { return a-b; }
};


template <size_t a_index,size_t b_index> struct Mul
{
  template <typename A,typename B>
  static auto eval(const A& a,const B& b) { return a*b; }
};


template <size_t a_index,size_t b_index> struct Div
{
  template <typename A,typename B>
  static auto eval(const A &a,const B &b) { return a/b; }
};


template <size_t x_index> struct Sqrt
{
  static float eval(float x) { return std::sqrt(x); }
};


template <size_t index> struct XValue {};
template <size_t index> struct YValue {};
template <size_t index> struct ZValue {};
template <size_t index,typename Expr> struct Node {};
template <size_t mat_index,size_t row,size_t col> struct Elem {};
template <typename Tag,size_t row,size_t col> struct VarElem {};

struct Empty {};
struct None {};

}


namespace {
template <size_t index,typename T>
struct IndexedValue {
  T value;
};
}


namespace {
template <typename First,typename Entry>
struct MapList : First, Entry {
};
}


template <typename First,size_t node_index,size_t adjoint_node_index>
static auto
  findAdjoint(
    MapList<First,MapEntry<node_index,adjoint_node_index>>,
    Indexed<node_index>
  )
{
  return Indexed<adjoint_node_index>{};
}


template <size_t node_index>
static auto findAdjoint(Empty,Indexed<node_index>)
{
  return None{};
}


template <
  typename First,
  size_t node_index1,
  size_t node_index2,
  size_t adjoint_node_index
>
static auto
  findAdjoint(
    MapList<First,MapEntry<node_index1,adjoint_node_index>>,
    Indexed<node_index2>
  )
{
  return findAdjoint(First{},Indexed<node_index2>{});
}


template <size_t result_index,size_t default_index>
static auto
  applyDefault(Indexed<result_index>,Indexed<default_index>)
{
  return Indexed<result_index>{};
}


template <size_t default_index>
static auto
  applyDefault(None,Indexed<default_index>)
{
  return Indexed<default_index>{};
}


template <
  typename Map,
  size_t node_index,
  size_t default_index
>
static auto
  findAdjoint(
    MapWithDefault<Map,default_index>,
    Indexed<node_index>
  )
{
  using Result = decltype(findAdjoint(Map{},Indexed<node_index>{}));
  return applyDefault(Result{},Indexed<default_index>{});
}


namespace {
template <size_t from_index,size_t to_index>
constexpr auto findNewIndex(MapEntry<from_index,to_index>)
{
  return Indexed<to_index>{};
}
}


template <size_t index,typename Adjoints>
static constexpr auto find_adjoint =
  decltype(findAdjoint(Adjoints{},Indexed<index>{}))::value;


namespace {
template <typename First,size_t index,typename T>
struct IndexedValueList : First, IndexedValue<index,T> {
  IndexedValueList(const First &first_arg,T value)
  : First(first_arg), IndexedValue<index,T>{value}
  {
  }
};
}


namespace {

template <typename Nodes,typename MapB> struct MergeResult {};

template <typename Nodes,typename MapB>
Nodes nodesOf(MergeResult<Nodes,MapB>);


template <typename Nodes,typename MapB>
MapB mapBOf(MergeResult<Nodes,MapB>);

}


namespace {
template <typename Tag,typename T>
struct Let {
  const T value;
};
}


namespace {
template <typename Tag,typename T>
struct Dual {
  T &value;
};
}


template <size_t index,typename Map>
static constexpr size_t mapped_index =
  decltype(findNewIndex<index>(Map{}))::value;


namespace {
template <typename Tag>
auto var()
{
  using Expr = Var<Tag>;
  using Node0 = Node<0,Expr>;
  return Graph<ScalarIndices<0>,List<Node0>>{};
}
}


namespace {
template <typename Tag>
auto constant()
{
  using Expr = Const<Tag>;
  using Node0 = Node<0,Expr>;
  return Graph<ScalarIndices<0>,List<Node0>>{};
}
}


template <typename Value>
static float
  floatValue(
    Graph<ScalarIndices<0>,
      List<
        Node<0,Const<Value>>
      >
    >
  )
{
  return Value::value();
}


namespace {
template <char...> struct Literal { };
}


template <char... chars>
static float
  floatValue(
    Graph<ScalarIndices<0>,
      List<
        Node<0,Literal<chars...>>
      >
    >
  )
{
  return valueOf(Literal<chars...>{});
}


template <size_t index,typename Nodes>
Graph<ScalarIndices<index>,Nodes>::operator float() const
{
  return floatValue(*this);
}


namespace {
template <typename A,typename Map>
auto mapExpr(Var<A>,Map)
{
  return Var<A>{};
}
}


namespace {
template <typename A,typename Map>
auto mapExpr(Const<A>,Map)
{
  return Const<A>{};
}
}


namespace {
template <char...chars,typename Map>
auto mapExpr(Literal<chars...>,Map)
{
  return Literal<chars...>{};
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
template <typename Expr>
None findNodeIndexHelper(...)
{
  return {};
}
}


namespace {
template <typename Expr,size_t index>
auto findNodeIndexHelper(const Node<index,Expr> &)
{
  return Indexed<index>{};
}
}


namespace {
template <
  typename Expr,
  typename... Nodes
>
auto findNodeIndex(Expr, List<Nodes...>)
{
  struct X : Nodes... {};
  return findNodeIndexHelper<Expr>(X{});
}
}


namespace {


template <typename NewMergedNodesArg,size_t new_index_arg>
struct UpdateMergedNodesResult {
};


template <typename NewNodes,size_t new_index>
static auto newNodesOf(UpdateMergedNodesResult<NewNodes,new_index>)
{
  return NewNodes{};
}


template <typename NewNodes,size_t new_index>
static constexpr size_t newIndexOf(UpdateMergedNodesResult<NewNodes,new_index>)
{
  return new_index;
}


}


namespace {
template <typename... Nodes,size_t new_index,typename Expr>
auto updateMergedNodes(List<Nodes...>,Indexed<new_index>,Expr)
{
  using NewNodes = List<Nodes...>;
  return UpdateMergedNodesResult<NewNodes,new_index>{};
}
}


namespace {
template <typename... Nodes,typename Expr>
auto updateMergedNodes(List<Nodes...>,None,Expr)
{
  static constexpr size_t new_index = sizeof...(Nodes);
  using NewNodes = List<Nodes...,Node<new_index,Expr>>;
  return UpdateMergedNodesResult<NewNodes,new_index>{};
}
}


namespace {
template <typename Nodes,size_t default_index>
struct DefaultableNodes {};
}


namespace {
template <typename Nodes,size_t default_index,size_t new_index,typename Expr>
auto
  updateMergedNodes(
    DefaultableNodes<Nodes,default_index>,
    Indexed<new_index>,
    Expr
  )
{
  using NewNodes = DefaultableNodes<Nodes,default_index>;
  return UpdateMergedNodesResult<NewNodes,new_index>{};
}
}


namespace {
template <typename Nodes,size_t default_index,typename Expr>
auto
  updateMergedNodes(
    DefaultableNodes<Nodes,default_index>,
    None,
    Expr
  )
{
  using NewNodes = DefaultableNodes<Nodes,default_index>;
  return UpdateMergedNodesResult<NewNodes,default_index>{};
}
}


namespace {
// If there are no more nodes to add, return what we've built.
template <typename NewMergedNodes,typename NewMapB>
auto buildMergedNodes(NewMergedNodes, List<>, NewMapB)
{
  return MergeResult<NewMergedNodes,NewMapB>{};
}
}


namespace {
template <typename Nodes,typename Expr>
auto insertNode(Nodes,Expr)
{
  using MaybeIndex = decltype(findNodeIndex(Expr{},Nodes{}));

  using UpdateResult = decltype(updateMergedNodes(Nodes{},MaybeIndex{},Expr{}));
    // This is a UpdateMergedNodesResult

  return UpdateResult{};
}
}


namespace {
template <typename Nodes,size_t default_index,typename Expr>
auto insertNode(DefaultableNodes<Nodes,default_index>,Expr)
{
  using MaybeIndex = decltype(findNodeIndex(Expr{},Nodes{}));

  using UpdateResult =
    decltype(
      updateMergedNodes(
        DefaultableNodes<Nodes,default_index>{},
        MaybeIndex{},
        Expr{}
      )
    );
    // This is a UpdateMergedNodesResult

  return UpdateResult{};
}
}


namespace {
// If we have nodes to add, add the first one and recurse.
template <
  typename MergedNodes,
  typename... BNodes,
  size_t index_b,
  typename ExprB,
  typename MapB
>
auto
  buildMergedNodes(
    MergedNodes,
    List<Node<index_b,ExprB>,BNodes...>,
    MapB
  )
{
  using UpdateResult =
    decltype(insertNode(MergedNodes{},mapExpr(ExprB{},MapB{})));

  using NewMergedNodes = decltype(newNodesOf(UpdateResult{}));
  constexpr auto new_index = newIndexOf(UpdateResult{});

  using NewMapB = MapList<MapB,MapEntry<index_b,new_index>>;
  return buildMergedNodes(NewMergedNodes{},List<BNodes...>{},NewMapB{});
}
}


namespace {
template <typename MergedNodes,typename...NodesB>
auto merge(MergedNodes,List<NodesB...>)
{
  using MapB = Empty;
  return buildMergedNodes(MergedNodes{},List<NodesB...>{},MapB{});
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
  return Graph<ScalarIndices<listSize<MergedNodes>>,decltype(new_nodes)>{};
}
}


namespace {
template <
  template <size_t,size_t> typename Op,
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto
  binary(
    Graph<ScalarIndices<index_a>,NodesA>,
    Graph<ScalarIndices<index_b>,NodesB>
  )
{
  constexpr size_t new_index_a = index_a;
  using MergeResult = decltype(merge(NodesA{},NodesB{}));
  using MergedNodes = decltype(nodesOf(MergeResult{}));
  using MapB        = decltype(mapBOf(MergeResult{}));
  constexpr size_t new_index_b = mapped_index<index_b,MapB>;
  return mergedGraph(MergedNodes{},Op<new_index_a,new_index_b>{});
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto
  operator+(
    Graph<ScalarIndices<index_a>,NodesA> graph_a,
    Graph<ScalarIndices<index_b>,NodesB> graph_b
  )
{
  return binary<Add>(graph_a,graph_b);
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto
  operator-(
    Graph<ScalarIndices<index_a>,NodesA> graph_a,
    Graph<ScalarIndices<index_b>,NodesB> graph_b
  )
{
  return binary<Sub>(graph_a,graph_b);
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto
  operator*(
    Graph<ScalarIndices<index_a>,NodesA> graph_a,
    Graph<ScalarIndices<index_b>,NodesB> graph_b
  )
{
  return binary<Mul>(graph_a,graph_b);
}
}


namespace {
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto
  operator/(
    Graph<ScalarIndices<index_a>,NodesA> graph_a,
    Graph<ScalarIndices<index_b>,NodesB> graph_b
  )
{
  return binary<Div>(graph_a,graph_b);
}
}


namespace {
template <typename Tag,typename T>
auto let(Graph<ScalarIndices<0>,List<Node<0,Var<Tag>>>>,const T& value)
{
  return Let<Tag,T>{value};
}
}


namespace {
template <size_t index,typename First,typename T>
auto valueList(const First &first,T value)
{
  return IndexedValueList<First,index,T>{first,value};
}
}


namespace {
template <typename Tag,typename T>
auto getLet2(const Let<Tag,T> &value)
{
  return value.value;
}
}


namespace {
template <typename Tag,typename Lets>
auto getLet(Tagged<Tag>,const Lets &lets)
{
  // This is searching through lets to find one that matches Let<Tag,T>
  // but our lets are of the form Let<Mat33Indices<...>,T>
  return getLet2<Tag>(lets);
}
}


namespace {
struct Mat33f {
  float values[3][3];

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
struct Vec3f {
  float x,y,z;

  Vec3f(float x_arg,float y_arg,float z_arg) : x(x_arg), y(y_arg), z(z_arg) {}

  bool operator==(const Vec3f &arg) const
  {
    return x==arg.x && y==arg.y && z==arg.z;
  }
};
}


static float xValue(const Vec3f &v) { return v.x; }
static float yValue(const Vec3f &v) { return v.y; }
static float zValue(const Vec3f &v) { return v.z; }


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


namespace {
template <size_t x_index,size_t y_index,size_t z_index>
struct Vec3Indices {};
}


namespace {
template <size_t x,size_t y,size_t z>
static auto indicesOf(Vec3Indices<x,y,z>)
{
  return Indices<x,y,z>{};
}
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto xValue(Graph<Vec3Indices<x,y,z>,Nodes>)
{
  return Graph<ScalarIndices<x>,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto yValue(Graph<Vec3Indices<x,y,z>,Nodes>)
{
  return Graph<ScalarIndices<y>,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto zValue(Graph<Vec3Indices<x,y,z>,Nodes>)
{
  return Graph<ScalarIndices<z>,Nodes>{};
}


namespace {
template <size_t x,size_t y,size_t z,typename Nodes>
struct Graph<Vec3Indices<x,y,z>,Nodes> {
  operator Vec3f() const;
};
}


static Vec3f vec3(float x,float y,float z)
{
  return Vec3f{x,y,z};
}


template <size_t xi,size_t yi,size_t zi,typename Nodes>
Graph<Vec3Indices<xi,yi,zi>,Nodes>::operator Vec3f() const
{
  float x = xValue(*this);
  float y = yValue(*this);
  float z = zValue(*this);
  return vec3(x,y,z);
}


template <size_t index,typename Map>
static auto mappedIndices(ScalarIndices<index>,Map)
{
  constexpr size_t new_index = mapped_index<index,Map>;
  return ScalarIndices<new_index>{};
}


template <size_t index,typename Map>
static auto adjointIndices(ScalarIndices<index>,Map)
{
  constexpr size_t new_index = find_adjoint<index,Map>;
  return ScalarIndices<new_index>{};
}


template <size_t x,size_t y,size_t z,typename Map>
static auto mappedIndices(Vec3Indices<x,y,z>,Map)
{
  constexpr size_t new_x = mapped_index<x,Map>;
  constexpr size_t new_y = mapped_index<y,Map>;
  constexpr size_t new_z = mapped_index<z,Map>;
  return Vec3Indices<new_x,new_y,new_z>{};
}


template <size_t x,size_t y,size_t z,typename Map>
static auto adjointIndices(Vec3Indices<x,y,z>,Map)
{
  constexpr size_t new_x = find_adjoint<x,Map>;
  constexpr size_t new_y = find_adjoint<y,Map>;
  constexpr size_t new_z = find_adjoint<z,Map>;
  return Vec3Indices<new_x,new_y,new_z>{};
}


namespace {
template <typename Row0,typename Row1,typename Row2>
struct Mat33Indices;
}


namespace {
template <size_t... args1,size_t... args2>
static auto merge(Indices<args1...>,Indices<args2...>)
{
  return Indices<args1...,args2...>{};
}
}


namespace {
template <size_t... args1,size_t... args2,size_t... args3>
static auto merge(Indices<args1...>,Indices<args2...>,Indices<args3...>)
{
  return Indices<args1...,args2...,args3...>{};
}
}


namespace {
template <typename Row0,typename Row1,typename Row2>
static auto indicesOf(Mat33Indices<Row0,Row1,Row2>)
{
  return merge(indicesOf(Row0{}),indicesOf(Row1{}),indicesOf(Row2{}));
}
}


template <typename Row0,typename Row1,typename Row2,typename Nodes>
static auto row(const Graph<Mat33Indices<Row0,Row1,Row2>,Nodes>,Indexed<0>)
{
  return Graph<Row0,Nodes>{};
}


template <typename Row0,typename Row1,typename Row2,typename Nodes>
static auto row(const Graph<Mat33Indices<Row0,Row1,Row2>,Nodes>,Indexed<1>)
{
  return Graph<Row1,Nodes>{};
}


template <typename Row0,typename Row1,typename Row2,typename Nodes>
static auto row(const Graph<Mat33Indices<Row0,Row1,Row2>,Nodes>,Indexed<2>)
{
  return Graph<Row2,Nodes>{};
}


// Maybe mappedIndices and adjointIndices this can be generalized somehow.
template <typename Row0,typename Row1,typename Row2,typename Map>
static auto mappedIndices(Mat33Indices<Row0,Row1,Row2>,Map)
{
  using NewRow0 = decltype(mappedIndices(Row0{},Map{}));
  using NewRow1 = decltype(mappedIndices(Row1{},Map{}));
  using NewRow2 = decltype(mappedIndices(Row2{},Map{}));
  return Mat33Indices<NewRow0,NewRow1,NewRow2>{};
}


template <size_t... indices,typename Map>
static auto mappedIndices(Indices<indices...>,Map)
{
  return Indices<mapped_index<indices,Map>...>{};
}


template <typename Row0,typename Row1,typename Row2,typename Map>
static auto adjointIndices(Mat33Indices<Row0,Row1,Row2>,Map)
{
  using NewRow0 = decltype(adjointIndices(Row0{},Map{}));
  using NewRow1 = decltype(adjointIndices(Row1{},Map{}));
  using NewRow2 = decltype(adjointIndices(Row2{},Map{}));
  return Mat33Indices<NewRow0,NewRow1,NewRow2>{};
}


template <size_t index>
static auto row(const Mat33f &m,Indexed<index>)
{
  float x = m.values[index][0];
  float y = m.values[index][1];
  float z = m.values[index][2];
  return vec3(x,y,z);
}


namespace {
template <typename Row0,typename Row1,typename Row2,typename Nodes>
struct Graph<Mat33Indices<Row0,Row1,Row2>,Nodes> {
  operator Mat33f() const
  {
    // Use other row function
    Vec3f row0 = row(*this,Indexed<0>{});
    Vec3f row1 = row(*this,Indexed<1>{});
    Vec3f row2 = row(*this,Indexed<2>{});
    return mat33(row0,row1,row2);
  }
};
}


static Vec3f vec3(const float (&v)[3])
{
  return Vec3f{v[0],v[1],v[2]};
}


static auto elem(const Vec3f &v,Indexed<0>) { return v.x; }
static auto elem(const Vec3f &v,Indexed<1>) { return v.y; }
static auto elem(const Vec3f &v,Indexed<2>) { return v.z; }


template <size_t r,size_t c,typename M>
static auto elem(const M &m)
{
  return elem(row(m,Indexed<r>{}),Indexed<c>{});
}


namespace {
template <typename Tag,size_t i,size_t j,typename Lets>
auto getLet(Tagged<VarElem<Tag,i,j>>,const Lets &lets)
{
  return elem<i,j>(getLet2<Tag>(lets));
}
}


namespace {
template <size_t index,typename T>
auto getValue2(const IndexedValue<index,T> &value)
{
  return value.value;
}
}


namespace {
template <size_t index,typename List>
auto getValue(Indexed<index>,const List &list)
{
  return getValue2<index>(list);
}
}


static bool debug = false;


namespace {
template <size_t index,size_t n_values>
void setValue3(ScalarIndices<index>,float value,float (&values)[n_values])
{
  if (debug) {
    cerr << "Setting values[" << index << "] to " << value << "\n";
  }

  if (index == n_values) {
    // We use index == n_values to mean that that the index is unused.
    return;
  }

  assert(index < n_values);
  values[index] = value;
}
}


namespace {
template <size_t index,size_t n_values>
auto getValue3(ScalarIndices<index>,const float (&values)[n_values])
{
  assert(index < n_values);
  return values[index];
}
}


namespace {
template <size_t n_values>
struct DefaultZero {
  const float (&values)[n_values];
};
}


namespace {
template <size_t index,size_t n_values>
auto getValue3(ScalarIndices<index>,const DefaultZero<n_values> &values)
{
  if (index == n_values) {
    return 0.0f;
  }

  assert(index < n_values);
  return values.values[index];
}
}


namespace {
template <size_t xi,size_t yi,size_t zi,size_t n_values>
void
  setValue3(
    Vec3Indices<xi,yi,zi>,
    const Vec3f &v,
    float (&values)[n_values]
  )
{
  setValue3(ScalarIndices<xi>{},v.x,values);
  setValue3(ScalarIndices<yi>{},v.y,values);
  setValue3(ScalarIndices<zi>{},v.z,values);
}
}


namespace {
template <size_t xi,size_t yi,size_t zi,typename Values>
auto getValue3(Vec3Indices<xi,yi,zi>,const Values &values)
{
  float x = getValue3(ScalarIndices<xi>{},values);
  float y = getValue3(ScalarIndices<yi>{},values);
  float z = getValue3(ScalarIndices<zi>{},values);
  return vec3(x,y,z);
}
}


namespace {
template <typename Row0,typename Row1,typename Row2,size_t n_values>
void
  setValue3(
    Mat33Indices<Row0,Row1,Row2>,
    const Mat33f &m,
    float (&values)[n_values]
  )
{
  setValue3(Row0{},vec3(m.values[0]),values);
  setValue3(Row1{},vec3(m.values[1]),values);
  setValue3(Row2{},vec3(m.values[2]),values);
}
}


namespace {
template <typename Row0,typename Row1,typename Row2,typename Values>
auto getValue3(Mat33Indices<Row0,Row1,Row2>,const Values &values)
{
  Vec3f row0 = getValue3(Row0{},values);
  Vec3f row1 = getValue3(Row1{},values);
  Vec3f row2 = getValue3(Row2{},values);
  return mat33(row0,row1,row2);
}
}


#if ADD_QR_DECOMP
namespace {
Vec3f operator-(const Vec3f &a,const Vec3f& b)
{
  return vec3(a.x-b.x,a.y-b.y,a.z-b.z);
}
}
#endif


#if ADD_QR_DECOMP
namespace {
Vec3f operator*(const Vec3f &a,float b)
{
  return vec3(a.x*b,a.y*b,a.z*b);
}
}
#endif


#if ADD_QR_DECOMP
namespace {
Vec3f operator/(const Vec3f &a,float b)
{
  return vec3(a.x/b,a.y/b,a.z/b);
}
}
#endif


namespace {
template <
  size_t xa,size_t ya,size_t za,
  size_t xb,size_t yb,size_t zb,
  typename NodesA,
  typename NodesB
>
auto
  operator-(
    Graph<Vec3Indices<xa,ya,za>,NodesA> a,
    Graph<Vec3Indices<xb,yb,zb>,NodesB> b
  )
{
  auto new_x = xValue(a)-xValue(b);
  auto new_y = yValue(a)-yValue(b);
  auto new_z = zValue(a)-zValue(b);

  return vec3(new_x,new_y,new_z);
}
}


namespace {
template <
  size_t x,size_t y,size_t z,
  size_t index_b,
  typename NodesA,
  typename NodesB
>
auto
  operator*(
    Graph<Vec3Indices<x,y,z>,NodesA> a,
    Graph<ScalarIndices<index_b>,NodesB> b
  )
{
  auto new_x = xValue(a)*b;
  auto new_y = yValue(a)*b;
  auto new_z = zValue(a)*b;

  return vec3(new_x,new_y,new_z);
}
}


namespace {
template <
  size_t x,size_t y,size_t z,
  size_t index_b,
  typename NodesA,
  typename NodesB
>
auto operator/(Graph<Vec3Indices<x,y,z>,NodesA> a,Graph<ScalarIndices<index_b>,NodesB> b)
{
  auto new_x = xValue(a)/b;
  auto new_y = yValue(a)/b;
  auto new_z = zValue(a)/b;

  return vec3(new_x,new_y,new_z);
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
template <typename A,typename Values,typename Lets>
auto evalExpr(Const<A>, const Values &,const Lets &)
{
  return A::value();
}
}


namespace {
template <char c>
float valueOf(Literal<c>)
{
  return c-'0';
}
}


namespace {
template <char... chars,typename Values,typename Lets>
auto evalExpr(Literal<chars...>, const Values &,const Lets &)
{
  return valueOf(Literal<chars...>{});
}
}


namespace {
template <typename Op,typename... Args>
auto evalOp(Op,Args... args)
{
  return Op::eval(args...);
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
auto evalNodes(Values values,List<Node<index,Expr>,Nodes...>,const Lets &lets)
{
  auto value = evalExpr(Expr{}, values, lets);
  return evalNodes(valueList<index>(values,value),List<Nodes...>{},lets);
}
}


namespace {
template <typename Nodes,typename... Lets>
static auto buildValues(Nodes,Lets... lets)
{
  struct : Lets... {} let_list {lets...};

  return evalNodes( /*values*/Empty{}, Nodes{}, let_list );
}
}


namespace {
template <typename Nodes,size_t index,typename ...Lets>
auto
  eval(
    Graph<ScalarIndices<index>, Nodes>,
    Lets... lets
  )
{
  auto values = buildValues(Nodes{},lets...);
  return getValue(Indexed<index>{},values);
}
}


namespace {
template <
  typename Nodes,
  size_t x_index,
  size_t y_index,
  size_t z_index,
  typename... Lets
>
auto
  eval(
    Graph<Vec3Indices<x_index,y_index,z_index>,Nodes>,
    Lets... lets
  )
{
  auto values = buildValues(Nodes{},lets...);
  float x = getValue(Indexed<x_index>{},values);
  float y = getValue(Indexed<y_index>{},values);
  float z = getValue(Indexed<z_index>{},values);
  return Vec3f{x,y,z};
}
}


namespace {
template <typename Nodes,size_t index,typename Values>
auto get(Graph<ScalarIndices<index>,Nodes>,const Values &values)
{
  return getValue(Indexed<index>{},values);
}
}


template <
  size_t x_index,size_t y_index,size_t z_index,
  typename XNodes,typename YNodes,typename ZNodes
>
static auto vec3(
  Graph<ScalarIndices<x_index>,XNodes>,
  Graph<ScalarIndices<y_index>,YNodes>,
  Graph<ScalarIndices<z_index>,ZNodes>
)
{
  constexpr size_t new_x_index = x_index;
  using XYMergeResult = decltype(merge(XNodes{},YNodes{}));
  using XYNodes = decltype(nodesOf(XYMergeResult{}));
  using MapY = decltype(mapBOf(XYMergeResult{}));
  constexpr size_t new_y_index = mapped_index<y_index,MapY>;
  using XYZMergeResult = decltype(merge(XYNodes{},ZNodes{}));
  using XYZNodes = decltype(nodesOf(XYZMergeResult{}));
  using MapZ = decltype(mapBOf(XYZMergeResult{}));
  constexpr size_t new_z_index = mapped_index<z_index,MapZ>;
  return Graph<Vec3Indices<new_x_index,new_y_index,new_z_index>,XYZNodes>{};
}


namespace {
template <typename Row0,typename Row1,typename Row2> struct Mat33Indices { };
}


namespace {
template <typename Tag>
auto
  let(
    Graph<Mat33Indices<
      Vec3Indices<0,1,2>,
      Vec3Indices<3,4,5>,
      Vec3Indices<6,7,8>
    >,
      List<
        Node<0,Var<VarElem<Tag,0,0>>>,
        Node<1,Var<VarElem<Tag,0,1>>>,
        Node<2,Var<VarElem<Tag,0,2>>>,
        Node<3,Var<VarElem<Tag,1,0>>>,
        Node<4,Var<VarElem<Tag,1,1>>>,
        Node<5,Var<VarElem<Tag,1,2>>>,
        Node<6,Var<VarElem<Tag,2,0>>>,
        Node<7,Var<VarElem<Tag,2,1>>>,
        Node<8,Var<VarElem<Tag,2,2>>>
      >
    >
    ,
    const Mat33f &value
  )
{
  return Let<Tag,Mat33f>{value};
}
}


namespace {
template <typename Tag>
auto mat33Var()
{
  return Graph<Mat33Indices<
    Vec3Indices<0,1,2>,
    Vec3Indices<3,4,5>,
    Vec3Indices<6,7,8>
  >,
    List<
      Node<0,Var<VarElem<Tag,0,0>>>,
      Node<1,Var<VarElem<Tag,0,1>>>,
      Node<2,Var<VarElem<Tag,0,2>>>,
      Node<3,Var<VarElem<Tag,1,0>>>,
      Node<4,Var<VarElem<Tag,1,1>>>,
      Node<5,Var<VarElem<Tag,1,2>>>,
      Node<6,Var<VarElem<Tag,2,0>>>,
      Node<7,Var<VarElem<Tag,2,1>>>,
      Node<8,Var<VarElem<Tag,2,2>>>
    >
  >{};
}
}


namespace {
template <
  typename Nodes,
  size_t r0xi,size_t r0yi,size_t r0zi,
  size_t r1xi,size_t r1yi,size_t r1zi,
  size_t r2xi,size_t r2yi,size_t r2zi,
  typename... Lets
>
auto
  eval(
    Graph<Mat33Indices<
      Vec3Indices<r0xi,r0yi,r0zi>,
      Vec3Indices<r1xi,r1yi,r1zi>,
      Vec3Indices<r2xi,r2yi,r2zi>
    >,
      Nodes
    >,
    Lets... lets
  )
{
  auto values = buildValues(Nodes{},lets...);
  float r0x = getValue(Indexed<r0xi>{},values);
  float r0y = getValue(Indexed<r0yi>{},values);
  float r0z = getValue(Indexed<r0zi>{},values);
  float r1x = getValue(Indexed<r1xi>{},values);
  float r1y = getValue(Indexed<r1yi>{},values);
  float r1z = getValue(Indexed<r1zi>{},values);
  float r2x = getValue(Indexed<r2xi>{},values);
  float r2y = getValue(Indexed<r2yi>{},values);
  float r2z = getValue(Indexed<r2zi>{},values);

  return
    mat33(
      vec3(r0x,r0y,r0z),
      vec3(r1x,r1y,r1z),
      vec3(r2x,r2y,r2z)
    );
}
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




template <
  typename R0,
  typename R1,
  typename R2,
  typename Row0Nodes,typename Row1Nodes,typename Row2Nodes
>
static auto
  mat33(
    Graph<R0,Row0Nodes>,
    Graph<R1,Row1Nodes>,
    Graph<R2,Row2Nodes>
  )
{
  using Row01MergeResult = decltype(merge(Row0Nodes{},Row1Nodes{}));
  using Row01Nodes = decltype(nodesOf(Row01MergeResult{}));
  using MapRow1 = decltype(mapBOf(Row01MergeResult{}));
  using Row012MergeResult = decltype(merge(Row01Nodes{},Row2Nodes{}));
  using Row012Nodes = decltype(nodesOf(Row012MergeResult{}));
  using MapRow2 = decltype(mapBOf(Row012MergeResult{}));
  using NewR0 = R0;
  using NewR1 = decltype(mappedIndices(R1{},MapRow1{}));
  using NewR2 = decltype(mappedIndices(R2{},MapRow2{}));

  return
    Graph<
      Mat33Indices<NewR0,NewR1,NewR2>,
      Row012Nodes
    >{};
}


template <typename Col0,typename Col1,typename Col2>
static auto columns(
  Col0 &col0,
  Col1 &col1,
  Col2 &col2
)
{
  auto m00 = xValue(col0);
  auto m01 = xValue(col1);
  auto m02 = xValue(col2);
  auto m10 = yValue(col0);
  auto m11 = yValue(col1);
  auto m12 = yValue(col2);
  auto m20 = zValue(col0);
  auto m21 = zValue(col1);
  auto m22 = zValue(col2);

  return mat33(
    vec3(m00,m01,m02),
    vec3(m10,m11,m12),
    vec3(m20,m21,m22)
  );
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
static auto elem(Graph<ScalarIndices<mat_index>,Nodes>)
{
  return mergedGraph(Nodes{},Elem<mat_index,row,col>{});
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Graph<Vec3Indices<x,y,z>,Nodes>,Indexed<0>)
{
  return Graph<ScalarIndices<x>,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Graph<Vec3Indices<x,y,z>,Nodes>,Indexed<1>)
{
  return Graph<ScalarIndices<y>,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Graph<Vec3Indices<x,y,z>,Nodes>,Indexed<2>)
{
  return Graph<ScalarIndices<z>,Nodes>{};
}


template <size_t row,size_t col>
static auto elem(const Mat33f &m)
{
  return m.values[row][col];
}


template <size_t index,typename M>
static auto col(const M &m)
{
  auto x = elem<0,index>(m);
  auto y = elem<1,index>(m);
  auto z = elem<2,index>(m);
  return vec3(x,y,z);
}


template <size_t index,typename Nodes>
static auto sqrt(Graph<ScalarIndices<index>,Nodes>)
{
  return mergedGraph(Nodes{},Sqrt<index>{});
}


template <typename V>
static auto mag(const V &v)
{
  return sqrt(dot(v,v));
}


template <typename Q,typename R>
struct QR {
  const Q q;
  const R r;
};


template <typename Q,typename R>
static QR<Q,R> qr(const Q &q,const R &r)
{
  return {q,r};
}


template <typename A>
static auto test3Func(const A &a)
{
  auto r = a;
  auto q = a*a;
  return qr(q,r);
}


namespace {
template <char... chars>
static auto operator""_t()
{
  return Graph<ScalarIndices<0>,List<Node<0,Literal<chars...>>>>{};
}
}


#if ADD_QR_DECOMP
template <typename A>
static auto qrDecomposition(const A &a)
{
  auto a1 = col<0>(a);
  auto a2 = col<1>(a);
  auto a3 = col<2>(a);
  auto u1 = a1;
  auto r11 = mag(u1);
  auto q1 = u1/r11;
  auto r12 = dot(a2,q1);
  auto u2 = a2 - q1*r12;
  auto r22 = mag(u2);
  auto q2 = u2/r22;
  auto r13 = dot(q1,a3);
  auto r23 = dot(q2,a3);
  auto u3 = a3 - q1*r13 - q2*r23;
  auto r33 = mag(u3);
  auto q3 = u3/r33;
  auto row0 = vec3(r11,r12,r13);
  auto row1 = vec3(0_t,r22,r23);
  auto row2 = vec3(0_t,0_t,r33);
  auto r = mat33(row0,row1,row2);
  auto q = columns(q1,q2,q3);
  return qr(q,r);
}
#endif


template <typename RandomEngine>
static float randomFloat(RandomEngine &engine)
{
  return std::uniform_real_distribution<float>(-1,1)(engine);
}


template <typename RandomEngine>
static Mat33f randomMat33(RandomEngine &engine)
{
  float m00 = randomFloat(engine);
  float m01 = randomFloat(engine);
  float m02 = randomFloat(engine);
  float m10 = randomFloat(engine);
  float m11 = randomFloat(engine);
  float m12 = randomFloat(engine);
  float m20 = randomFloat(engine);
  float m21 = randomFloat(engine);
  float m22 = randomFloat(engine);

  const float values[3][3] = {
    {m00,m01,m02},
    {m10,m11,m12},
    {m20,m21,m22}
  };

  return {values};
}


namespace {


template <typename Adjoints,typename Nodes>
struct AdjointGraph {
};


template <typename Adjoints,typename Nodes>
static constexpr auto adjointsOf(AdjointGraph<Adjoints,Nodes>)
{
  return Adjoints{};
}


template <typename Adjoints,typename Nodes>
static auto nodesOf(AdjointGraph<Adjoints,Nodes>)
{
  return Nodes{};
}


}


template <typename OldAdjointList,size_t adjoint_index,size_t value_index>
static auto
  setAdjoint(OldAdjointList,Indexed<adjoint_index>,Indexed<value_index>)
{
  return MapList<OldAdjointList,MapEntry<adjoint_index,value_index>>{};
}


// adjoints[i] += k
template <typename Adjoints,typename... Nodes,size_t i,size_t k>
static auto addTo(AdjointGraph<Adjoints,List<Nodes...>>,Indexed<i>,Indexed<k>)
{
  constexpr size_t adjoint_i = find_adjoint<i,Adjoints>;

  using InsertResult =
    decltype(insertNode(List<Nodes...>{},Add<adjoint_i,k>{}));

  using NewNodes = decltype(newNodesOf(InsertResult{}));
  constexpr size_t new_index = newIndexOf(InsertResult{});

  using NewAdjoints =
    decltype(setAdjoint(Adjoints{},Indexed<i>{},Indexed<new_index>{}));

  return AdjointGraph<NewAdjoints,NewNodes>{};
}


// adjoints[i] -= k
template <typename Adjoints,typename... Nodes,size_t i,size_t k>
static auto subFrom(AdjointGraph<Adjoints,List<Nodes...>>,Indexed<i>,Indexed<k>)
{
  constexpr size_t adjoint_i = find_adjoint<i,Adjoints>;

  using InsertResult =
    decltype(insertNode(List<Nodes...>{},Sub<adjoint_i,k>{}));

  using NewNodes = decltype(newNodesOf(InsertResult{}));
  constexpr size_t new_index = newIndexOf(InsertResult{});

  using NewAdjoints =
    decltype(setAdjoint(Adjoints{},Indexed<i>{},Indexed<new_index>{}));

  return AdjointGraph<NewAdjoints,NewNodes>{};
}


// adjoints[i] += expr;
template <typename Adjoints,typename Nodes,size_t i,typename Expr>
static auto addExpr(AdjointGraph<Adjoints,Nodes>,Indexed<i>,Expr)
{
  using InsertResult1 = decltype(insertNode(Nodes{},Expr{}));
  using NewNodes1 = decltype(newNodesOf(InsertResult1{}));
  constexpr size_t expr = newIndexOf(InsertResult1{});

  // add adjoints[k]*j to adjoints[i]
  return
    addTo(
      AdjointGraph<Adjoints,NewNodes1>{},
      Indexed<i>{},
      Indexed<expr>{}
    );
}


// adjoints[i] -= expr;
template <typename Adjoints,typename Nodes,size_t i,typename Expr>
static auto subExpr(AdjointGraph<Adjoints,Nodes>,Indexed<i>,Expr)
{
  using InsertResult1 = decltype(insertNode(Nodes{},Expr{}));
  using NewNodes1 = decltype(newNodesOf(InsertResult1{}));
  constexpr size_t expr = newIndexOf(InsertResult1{});

  // add adjoints[k]*j to adjoints[i]
  return
    subFrom(
      AdjointGraph<Adjoints,NewNodes1>{},
      Indexed<i>{},
      Indexed<expr>{}
    );
}


template <typename AdjointGraph,size_t k,typename Tag>
static auto addDeriv(AdjointGraph,Node<k,Var<Tag>>)
{
  return AdjointGraph{};
}


template <typename AdjointGraph,size_t k,typename Tag>
static auto addDeriv(AdjointGraph,Node<k,Const<Tag>>)
{
  return AdjointGraph{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i,size_t j>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Add<i,j>>)
{
  // adjoints[i] += adjoints[k];
  // adjoints[j] += adjoints[k];
  //
  constexpr size_t adjoint_k = find_adjoint<k,Adjoints>;

  using NewGraph1 =
    decltype(
      addTo(AdjointGraph<Adjoints,Nodes>{},Indexed<i>{},Indexed<adjoint_k>{})
    );

  // Add a new node which is the sum of node[adjoints[j]] and node[adjoints[k]]
  // Replace adjoints[j] with the new node index.
  using NewGraph2 =
    decltype(addTo(NewGraph1{},Indexed<j>{},Indexed<adjoint_k>{}));

  return NewGraph2{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i,size_t j>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Sub<i,j>>)
{
  // adjoints[i] += adjoints[k];
  // adjoints[j] -= adjoints[k];
  //
  constexpr size_t adjoint_k = find_adjoint<k,Adjoints>;

  using NewGraph1 =
    decltype(
      addTo(AdjointGraph<Adjoints,Nodes>{},Indexed<i>{},Indexed<adjoint_k>{})
    );

  // Add a new node which is the sum of node[adjoints[j]] and node[adjoints[k]]
  // Replace adjoints[j] with the new node index.
  using NewGraph2 =
    decltype(subFrom(NewGraph1{},Indexed<j>{},Indexed<adjoint_k>{}));

  return NewGraph2{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Sqrt<i>>)
{
  // adjoints[i] += adjoints[k]*0.5/sqrt(i)
  using SqrtI = decltype(insertNode(Nodes{},Sqrt<i>{}));
  using Nodes2 = decltype(newNodesOf(SqrtI{}));
  constexpr size_t sqrt_i = newIndexOf(SqrtI{});
  using CHalf = decltype(insertNode(Nodes2{},Const<Half>{}));
  constexpr size_t half = newIndexOf(CHalf{});
  using Nodes3 = decltype(newNodesOf(CHalf{}));
  using Coef = decltype(insertNode(Nodes3{},Div<half,sqrt_i>{}));
  using Nodes4 = decltype(newNodesOf(Coef{}));
  constexpr size_t coef = newIndexOf(Coef{});

  constexpr size_t adjoint_k = find_adjoint<k,Adjoints>;
  return
    addExpr(
      AdjointGraph<Adjoints,Nodes4>{},
      Indexed<i>{},
      Mul<adjoint_k,coef>{}
    );
}


template <typename Adjoints,typename Nodes,size_t k,size_t i,size_t j>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Mul<i,j>>)
{
  constexpr size_t adjoint_k = find_adjoint<k,Adjoints>;
  using NewGraph0 = AdjointGraph<Adjoints,Nodes>;

  // adjoints[i] += adjoints[k]*j
  using NewGraph1 =
    decltype(addExpr(NewGraph0{}, Indexed<i>{}, Mul<adjoint_k,j>{}));

  // adjoints[j] += adjoints[k]*i;
  using NewGraph2 =
    decltype(addExpr(NewGraph1{}, Indexed<j>{}, Mul<adjoint_k,i>{}));

  return NewGraph2{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i,size_t j>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Div<i,j>>)
{
// d(a/b) = (b*d(a) - a*d(b))/(b**2)
//        = d(a)/b - a*d(b)/(b**2)
// d(a) += dresult/b
// d(b) += dresult*-a/(b**2)

  constexpr size_t adjoint_k = find_adjoint<k,Adjoints>;

  // adjoints[i] += adjoints[k]/j
  using NewGraph1 =
    decltype(
      addExpr(
        AdjointGraph<Adjoints,Nodes>{},
        Indexed<i>{},
        Div<adjoint_k,j>{}
      )
    );

  using Adjoints2 = decltype(adjointsOf(NewGraph1{}));
  // adjoints[j] += adjoints[k]*i/(j**2);

  using J2 = decltype(insertNode(nodesOf(NewGraph1{}),Mul<j,j>{}));
  constexpr size_t j2 = newIndexOf(J2{});
  using Nodes2 = decltype(newNodesOf(J2{}));

  using IDIVJ2 = decltype(insertNode(Nodes2{},Div<i,j2>{}));
  constexpr size_t i_div_j2 = newIndexOf(IDIVJ2{});
  using Nodes3 = decltype(newNodesOf(IDIVJ2{}));

  return
    subExpr(
      AdjointGraph<Adjoints2,Nodes3>{},Indexed<j>{},Mul<adjoint_k,i_div_j2>{}
    );
}


template <char... chars,typename Adjoints,typename Nodes,size_t k>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Literal<chars...>>)
{
  return AdjointGraph<Adjoints,Nodes>{};
}


template <typename Adjoints,typename NewNodes>
static auto revNodes(AdjointGraph<Adjoints,NewNodes>,List<>)
{
  return AdjointGraph<Adjoints,NewNodes>{};
}


// revNodes adds derivatives to nodes and updates the adjoint list.
// returns an AdjointGraph<Adjoints,NodeList>
template <
  typename AdjointGraph,
  size_t index,
  typename Expr,
  typename...Nodes
>
static auto
  revNodes(
    AdjointGraph,
    List<Node<index,Expr>,Nodes...>
  )
{
  // We'll need to have an array which indicates what the adjoint node
  // is for each node.  This array will be updated as we process through
  // the nodes in reverse.
  auto newgraph = revNodes(AdjointGraph{},List<Nodes...>{});
  return addDeriv( newgraph, Node<index,Expr>{} );
}


template <size_t zero_node,typename ResultAdjointMap>
static auto
  makeAdjoints(
    Indexed</*node_count*/0>,
    Indexed<zero_node>,
    ResultAdjointMap
  )
{
  return Empty{};
}


template <size_t node_count,size_t zero_node,typename ResultAdjointMap>
static auto
  makeAdjoints(Indexed<node_count>,Indexed<zero_node>,ResultAdjointMap)
{
  using First =
    decltype(
      makeAdjoints(
        Indexed<node_count-1>{},
        Indexed<zero_node>{},
        ResultAdjointMap{}
      )
    );

  constexpr size_t x =
    decltype(findAdjoint(ResultAdjointMap{},Indexed<node_count-1>{}))::value;

  return MapList<First,MapEntry<node_count-1,x>>{};
}


// Create the nodes that contain the adjoints by going through a forward
// and reverse pass.  In the forward pass, we introduce an adjoint node
// for each node.  In the reverse pass, we update the adjoint nodes.
template <
  typename... Nodes,
  typename ResultIndices,
  typename ResultAdjointMap
>
static auto adjointNodes(ResultIndices,ResultAdjointMap,List<Nodes...>)
{
  // Add a zero node to the nodes if we don't have one, since we'll need
  // to initialize all the adjoints to this.
  using InsertResult = decltype(insertNode(List<Nodes...>{},Const<Zero>{}));
  using NodesWithDResult = decltype(newNodesOf(InsertResult{}));
  constexpr size_t zero_index = newIndexOf(InsertResult{});

  using Adjoints2 =
    decltype(
      makeAdjoints(
        Indexed<sizeof...(Nodes)>{},
        Indexed<zero_index>{},
        MapWithDefault<ResultAdjointMap,zero_index>{}
      )
    );

  // Process the nodes in reverse, adding new nodes and updating the adjoints.
  // If we encounter a node which has adjoint values, then those need to
  // be added in.
  using RevResult =
    decltype(
      revNodes(
        AdjointGraph<Adjoints2,NodesWithDResult>{},
        List<Nodes...>{}
      )
    );

  using NewNodes = decltype(nodesOf(RevResult{}));
  using NewAdjoints = decltype(adjointsOf(RevResult{}));
  return AdjointGraph<NewAdjoints,NewNodes>{};
}



static void testFindAdjoint()
{
  using Adjoints =
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<
    MapList<Empty,
    MapEntry<0, 5>>,
    MapEntry<1, 5>>,
    MapEntry<2, 5>>,
    MapEntry<3, 5>>,
    MapEntry<4, 5>>,
    MapEntry<4, 6>>,
    MapEntry<2, 8>>,
    MapEntry<3, 10>>,
    MapEntry<0, 12>>, // dA = C*B
    MapEntry<1, 14>>;

  constexpr size_t result =
    decltype(findAdjoint(Adjoints{},Indexed<0>{}))::value;

  static_assert(result == 12);
}


static void testAddDeriv()
{
  // Test adding a multiplication derivative to the adjoints.

  struct A;
  struct B;

  using Nodes =
    List<
      Node<0,Var<A>>,
      Node<1,Var<B>>,
      Node<2,Mul<0,1>>,
      Node<3,Const<Zero>>,
      Node<4,Const<One>>
    >;

  using Adjoints =
    MapList<
      MapList<
        MapList<Empty,MapEntry<0,3>>,  // adjoints[0] = 3
        MapEntry<1,3>                      // adjoints[1] = 3
      >,
      MapEntry<2,4>                        // adjoints[2] = 4
    >;

  using Graph = AdjointGraph<Adjoints,Nodes>;
  constexpr size_t i = 0;
  constexpr size_t j = 1;
  constexpr size_t k = 2;
  // nk = ni*nj;
  // ai += ak*nj;
  // aj += ak*ni;
  using NodeParam = Node<k,Mul<i,j>>;
  using Result = decltype(addDeriv(Graph{},NodeParam{}));

  using ExpectedNodes =
    List<
      Node<0,Var<A>>,
      Node<1,Var<B>>,
      Node<2,Mul<0,1>>,
      Node<3,Const<Zero>>,
      Node<4,Const<One>>,
      Node<5,Mul<4,1>>, // 1*A
      Node<6,Add<3,5>>, // 0 + 1*A
      Node<7,Mul<4,0>>, // 1*B
      Node<8,Add<3,7>>  // 0 + 1*B
    >;

  using ExpectedAdjoints =
    MapList<
      MapList<
        MapList<
          MapList<
            MapList<Empty,MapEntry<0,3>>,  // adjoints[0] = 3
            MapEntry<1,3>                      // adjoints[1] = 3
          >,
          MapEntry<2,4>                        // adjoints[2] = 4
        >,
        MapEntry<0,6>                          // adjoints[0] = 6 // 0 + 1*B
      >,
      MapEntry<1,8>                            // adjoints[1] = 8 // 0 + 1*A
    >;

  using Expected = AdjointGraph<ExpectedAdjoints,ExpectedNodes>;
  static_assert(is_same_v<Result,Expected>);
}


template <
  typename Output,
  typename Nodes,
  typename Value,
  size_t n_nodes,
  typename AdjointNodes
>
static void
  setValue2(
    Graph<Output,Nodes>,
    const Value& value,
    float (&values)[n_nodes],
    AdjointNodes
  )
{
  // By using DefaultableNodes here, if there is no match for an expr
  // during the merge, we get an index of n_nodes and the nodes aren't
  // changed, instead of the usual behavior where we add a new node and
  // use its index.
  using MergeResult =
    decltype(merge(DefaultableNodes<AdjointNodes,n_nodes>{},Nodes{}));

  using Map = decltype(mapBOf(MergeResult{}));
  setValue3(mappedIndices(Output{},Map{}),value,values);
}


namespace {

// Done
template <
  size_t begin,
  size_t end,
  size_t n_values,
  typename Nodes
>
auto
  evaluate(
    Nodes,
    float (&/*values*/)[n_values]
  ) -> std::enable_if_t<begin==end>
{
}


// Skip evaluation until we get to the begin node
template <
  size_t begin,
  size_t end,
  size_t first,
  size_t n_values,
  typename Expr,
  typename... Nodes
>
auto
  evaluate(
    List<Node<first,Expr>,Nodes...>,
    float (&values)[n_values]
  ) -> std::enable_if_t<(first < begin)>
{
  evaluate<begin,end>(List<Nodes...>{},values);
}


// Evaluate the first node and then evaluate the rest.
template <
  size_t begin,
  size_t end,
  size_t n_values,
  typename Expr,
  typename... Nodes
>
auto
  evaluate(
    List<Node<begin,Expr>,Nodes...>,
    float (&values)[n_values]
  ) -> std::enable_if_t<(begin < end)>
{
  static_assert(begin < n_values);
  setValue(values,Node<begin,Expr>{});

  if (debug) {
    cerr << "Set node " << begin << " to " << values[begin] << "\n";
  }

  evaluate<begin+1,end>(List<Nodes...>{},values);
}

}



namespace {

// Variable values are set externally, so there's nothing to do here.
// Not sure if this is the right thing to do.  Might be better to have
// a let list instead.
template <size_t n,size_t index,typename A>
void setValue(float (&/*values*/)[n],Node<index,Var<A>>)
{
}


template <size_t n,size_t index,typename A>
void setValue(float (&values)[n],Node<index,Const<A>>)
{
  values[index] = A::value();
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Add<a,b>>)
{
  values[index] = values[a] + values[b];
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Sub<a,b>>)
{
  values[index] = values[a] - values[b];
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Mul<a,b>>)
{
  values[index] = values[a] * values[b];
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Div<a,b>>)
{
  values[index] = values[a] / values[b];
}


template <size_t n,size_t index,size_t a>
void setValue(float (&values)[n],Node<index,Sqrt<a>>)
{
  values[index] = sqrt(values[a]);
}

template <size_t n,size_t index,char... chars>
void setValue(float (&values)[n],Node<index,Literal<chars...>>)
{
  values[index] = valueOf(Literal<chars...>{});
}

}


namespace {
template <size_t index>
auto indicesOf(ScalarIndices<index>)
{
  return Indices<index>{};
}
}


namespace {
template <typename Result>
static auto buildMap(Result,Indices<>,Indices<>)
{
  return Result{};
}
}


namespace {
template <typename Result,size_t index,size_t adjoint_index,size_t... indices,size_t... adjoint_indices>
static auto
  buildMap(
    Result,
    Indices<index,indices...>,
    Indices<adjoint_index,adjoint_indices...>
  )
{
  return
    buildMap(
      MapList<Result,MapEntry<index,adjoint_index>>{},
      Indices<indices...>{},
      Indices<adjoint_indices...>{}
    );
}
}


namespace {
template <typename Indices,typename AdjointIndices>
static auto makeMap(Indices,AdjointIndices)
{
  return buildMap(Empty{},Indices{},AdjointIndices{});
}
}


namespace {
template <
  size_t index1,
  size_t index2,
  size_t adjoint_index1,
  size_t adjoint_index2
>
static auto
  makeMap(Indices<index1,index2>,Indices<adjoint_index1,adjoint_index2>)
{
  return
    MapList<
      MapList<
        Empty,
        MapEntry<index1,adjoint_index1>
      >,
      MapEntry<index2,adjoint_index2>
    >{};
}
}


namespace {
template <typename Graph, typename AdjointGraph> struct Function;

template <
  typename Output,
  typename ValueNodes,
  typename AdjointOutput,
  typename AdjointValueNodes
>
struct
  Function<
    Graph<Output,ValueNodes>,
    Graph<AdjointOutput,AdjointValueNodes>
  >
{
  using ValueGraph = Graph<Output,ValueNodes>;

  // Merge the output graph and the adjoint output graph so that
  // we have a common set of nodes to work with.
  using MergeResult = decltype(merge(ValueNodes{},AdjointValueNodes{}));
  using CommonOutputNodes = decltype(nodesOf(MergeResult{}));

  using MappedAdjointValueIndices =
    decltype(
      ::mappedIndices(indicesOf(AdjointOutput{}),mapBOf(MergeResult{}))
    );

  using AdjointGraph =
    decltype(
      adjointNodes(
        indicesOf(Output{}),
        makeMap(
          indicesOf(Output{}),
          MappedAdjointValueIndices{}
        ),
        CommonOutputNodes{}
      )
    );

  using AdjointNodes = decltype(nodesOf(AdjointGraph{}));
  using Adjoints = decltype(adjointsOf(AdjointGraph{}));

  // Given a Graph, find the indices of the nodes in this function's
  // graph which correspond to the nodes in the given graph.
  template <typename OutputArg,typename Nodes>
  static auto mappedIndices(Graph<OutputArg,Nodes>)
  {
    using MergeResult = decltype(merge(AdjointNodes{},Nodes{}));
    using Map = decltype(mapBOf(MergeResult{}));
    return ::mappedIndices(OutputArg{},Map{});
  }

  template <typename G>
  static constexpr size_t derivIndex(G)
  {
    constexpr size_t index = mappedIndex(G{});
    return findAdjoint(Adjoints{},Indexed<index>{});
  }

  template <typename G>
  static auto derivIndices(G)
  {
    using MappedIndices = decltype(mappedIndices(G{}));
    constexpr size_t null_index = n_values;

    return
      ::adjointIndices(
        MappedIndices{},
        MapWithDefault<Adjoints,null_index>{}
      );
  }

  static constexpr size_t dresult_index = derivIndex(ValueGraph{});
  static constexpr size_t n_values = sizeOf(AdjointNodes{});
  float values[n_values];

  template <typename Graph,typename T>
  void set(Graph,const T& value)
  {
    setValue2(Graph{},value,values,AdjointNodes{});
  }

  void evaluate()
  {
    constexpr size_t n_value_nodes = listSize<ValueNodes>;
    ::evaluate<0,n_value_nodes>(AdjointNodes{},values);
  }

  template <typename OutputArg,typename NodesArg>
  auto get(Graph<OutputArg,NodesArg>) const
  {
    return getValue3(mappedIndices(Graph<OutputArg,NodesArg>{}),values);
  }

  void evaluateDerivs()
  {
    constexpr size_t n_value_nodes = listSize<ValueNodes>;
    if (debug) {
      cerr << "Evaluating nodes " << n_value_nodes << " up to " <<
        n_values << "\n";
    }
    ::evaluate<n_value_nodes,n_values>(AdjointNodes(),values);
  }

  template <typename Graph>
  auto getDeriv(Graph) const
  {
    using DerivIndices = decltype(derivIndices(Graph{}));
    // We need to be able to get derivatives for values that aren't in the
    // graph.  In this case, we should return 0.  To make this work, we
    // have a special DefaultZero type.
    return getValue3(DerivIndices{},DefaultZero<n_values>{values});
  }
};
}



static void testMulFunction()
{
  auto a = var<struct A>();
  auto b = var<struct B>();
  auto c = a*b;
  auto dc = var<struct DC>();
  float a_val = 5;
  float b_val = 6;
  Function< decltype(c), decltype(dc) > f;
  f.set(a,a_val);
  f.set(b,b_val);
  f.evaluate();
  assert(f.get(c) == a_val*b_val);
  f.set(dc,1);
  f.evaluateDerivs();
  float da = f.getDeriv(a);
  float db = f.getDeriv(b);
  assert(da == 6);
  assert(db == 5);
}


static void testDotFunction()
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
  auto dc = var<struct DC>();
  Function< decltype(c), decltype(dc) > f;
  Vec3f a_val = {1,2,3};
  Vec3f b_val = {4,5,6};
  f.set(ax,a_val.x);
  f.set(ay,a_val.y);
  f.set(az,a_val.z);
  f.set(bx,b_val.x);
  f.set(by,b_val.y);
  f.set(bz,b_val.z);
  f.evaluate();
  float value = f.get(c);
  assert(value == dot(a_val,b_val));
  f.set(dc,1);
  f.evaluateDerivs();
  float dax = f.getDeriv(ax);
  float day = f.getDeriv(ay);
  float daz = f.getDeriv(az);
  float dbx = f.getDeriv(bx);
  float dby = f.getDeriv(by);
  float dbz = f.getDeriv(bz);
  Vec3f da_val = {dax,day,daz};
  Vec3f db_val = {dbx,dby,dbz};
  assert(da_val == vec3(4,5,6));
  assert(db_val == vec3(1,2,3));
}


template <typename F>
static float finiteDeriv(const F &f,float &var,float h = 1e-3)
{
  float old_value = var;
  var = old_value - h;
  float value1 = f();
  var = old_value + h;
  float value2 = f();
  var = old_value;
  return (value2 - value1)/(2*h);
}


static void assertNear(float actual,float expected,float tolerance)
{
  float delta = fabsf(actual-expected);
  if (delta <= tolerance) return;
  cerr << "actual:    " << actual << "\n";
  cerr << "expected:  " << expected << "\n";
  cerr << "delta:     " << delta << "\n";
  cerr << "tolerance: " << tolerance << "\n";
  assert(false);
}


static void testDivFunction()
{
  auto a = var<struct A>();
  auto b = var<struct B>();
  auto c = a/b;
  auto dc = var<struct DC>();
  float a_val = 5;
  float b_val = 6;
  float dc_val = 0.5;
  Function< decltype(c), decltype(dc) > f;
  f.set(a,a_val);
  f.set(b,b_val);
  f.evaluate();
  assert(f.get(c) == a_val/b_val);
  f.set(dc,dc_val);
  f.evaluateDerivs();
  float da = f.getDeriv(a);
  float db = f.getDeriv(b);
  assert(da == 1/6.0f*dc_val);
  float expected_db = -5/float(6*6)*dc_val;
  assert(db == expected_db);
  auto f_ab = [&]{ return a_val/b_val; };
  float fda = finiteDeriv(f_ab,a_val)*dc_val;
  float fdb = finiteDeriv(f_ab,b_val)*dc_val;
  assertNear(da,fda,1e-4);
  assertNear(db,fdb,1e-4);
}


static void testSqrtFunction()
{
  auto a = var<struct A>();
  auto b = sqrt(a);
  float a_val = 25;
  auto db = var<struct DB>();
  Function< decltype(b), decltype(db) > f;
  f.set(a,a_val);
  f.evaluate();
  assert(f.get(b) == 5);
  float db_val = 0.5;
  f.set(db,db_val);
  f.evaluateDerivs();
  float da = f.getDeriv(a);
  float expected_da = 0.5/sqrt(a_val)*db_val;
  assert(da == expected_da);

  auto f_a = [&]{ return sqrt(a_val); };
  assertNear(finiteDeriv(f_a,a_val)*db_val,da,0.002);
}


static void testMagFunction()
{
  auto vx = var<struct VX>();
  auto vy = var<struct VY>();
  auto vz = var<struct VZ>();
  auto v = vec3(vx,vy,vz);
  auto a = mag(v);
  auto da = var<struct DA>();

  float vx_val = 1;
  float vy_val = 2;
  float vz_val = 3;

  Vec3f v_val = vec3(vx_val,vy_val,vz_val);
  Function< decltype(a), decltype(da) > f;

  f.set(v,v_val);
  f.evaluate();
  float mag_v = mag(v_val);
  float a_val = f.get(a);
  assertNear(a_val, mag_v, 0);
  float da_val = 0.5;
  f.set(da,da_val);
  f.evaluateDerivs();
  Vec3f dv = f.getDeriv(v);

  auto f_v = [&]{ return mag(v_val); };
  float tolerance = 0.002;
  float fdvx = finiteDeriv(f_v, v_val.x)*da_val;
  float fdvy = finiteDeriv(f_v, v_val.y)*da_val;
  float fdvz = finiteDeriv(f_v, v_val.z)*da_val;
  assertNear(fdvx,dv.x,tolerance);
  assertNear(fdvy,dv.y,tolerance);
  assertNear(fdvz,dv.z,tolerance);
}


template <typename Q,typename R>
static auto indicesOf(QR<Q,R>)
{
  return merge(indicesOf(Q{}),indicesOf(R{}));
}


template <typename Q,typename R,typename Values>
static auto getValue3(QR<Q,R>,const Values &values)
{
  auto q = getValue3(Q{},values);
  auto r = getValue3(R{},values);
  return qr(q,r);
}


static void test3()
{
  auto a = var<struct A>();
  auto qr = test3Func(a);
  auto dq = var<struct DQ>();
  auto dr = var<struct DR>();
  auto dqr = ::qr(dq,dr);
  float dr_val = 1;
  float dq_val = 0;

  // qr is a QR<Q,R>, where Q and R are graphs with their own nodes.
  // It needs to be that way so that the qrDecomposition() function is
  // agnostic to the representation of the result.
  using QRMergeResult = decltype(merge(nodesOf(qr.q),nodesOf(qr.r)));
  using QRNodes = decltype(nodesOf(QRMergeResult{}));
  using RMap = decltype(mapBOf(QRMergeResult{}));
  using QIndices = decltype(outputOf(qr.q));
  using RIndices = decltype(mappedIndices(outputOf(qr.r), RMap{}));
  using DQRMergeResult = decltype(merge(nodesOf(dqr.q),nodesOf(dqr.r)));
  using DQRNodes = decltype(nodesOf(DQRMergeResult{}));
  using DRMap = decltype(mapBOf(DQRMergeResult{}));
  using DQIndices = decltype(outputOf(dqr.q));
  using DRIndices = decltype(mappedIndices(outputOf(dqr.r), DRMap{}));
  auto q = Graph<QIndices,QRNodes>{};
  auto r = Graph<RIndices,QRNodes>{};

  Function<
    Graph<QR<QIndices,RIndices>,QRNodes>,
    Graph<QR<DQIndices,DRIndices>,DQRNodes>
  > f;

  float a_val = 4;

  f.set(a,a_val);
  f.evaluate();
  float q_val = f.get(q);
  float r_val = f.get(r);
  auto expected_qr = test3Func(a_val);
  assertNear(q_val, expected_qr.q, 0);
  assert(r_val == expected_qr.r);
  f.set(dq, dq_val);
  f.set(dr, dr_val);
  f.evaluateDerivs();

  float da = f.getDeriv(a);

  // Verify the derivatives.
  float sum = 0;

  {
    auto q_a = [&]{ return test3Func(a_val).q; };
    sum += finiteDeriv(q_a, a_val) * dq_val;
  }

  {
    auto r_a = [&]{ return test3Func(a_val).r; };
    sum += finiteDeriv(r_a, a_val) * dr_val;
  }

  assertNear(da,sum,3e-4);
}


#if 0
static void show(const char *name,const Mat33f &value)
{
  cerr << name << ": ";

  for (int i=0; i!=3; ++i) {
    for (int j=0; j!=3; ++j) {
      cerr << " " << value.values[i][j];
    }
    cerr << "\n";
  }
}
#endif



#if ADD_QR_DECOMP
static void testQRDecompFunction()
{
  std::mt19937 engine(/*seed*/1);
  auto a = mat33Var<struct A>();
  auto qr = qrDecomposition(a);
  auto dr = mat33Var<struct DR>();
  auto dq = mat33Var<struct DQ>();
  auto dqr = ::qr(dq,dr);
  Mat33f a_val = randomMat33(engine);
  Mat33f dr_val = randomMat33(engine);
  Mat33f dq_val = randomMat33(engine);

  // qr is a QR<Q,R>, where Q and R are graphs with their own nodes.
  // It needs to be that way so that the qrDecomposition() function is
  // agnostic to the representation of the result.
  using MergeResult = decltype(merge(nodesOf(qr.q),nodesOf(qr.r)));
  using QRNodes = decltype(nodesOf(MergeResult{}));
  using QIndices = decltype(outputOf(qr.q));
  using RMap = decltype(mapBOf(MergeResult{}));
  using RIndices = decltype(mappedIndices(outputOf(qr.r), RMap{}));
  using DQRMergeResult = decltype(merge(nodesOf(dqr.q),nodesOf(dqr.r)));
  using DRMap = decltype(mapBOf(DQRMergeResult{}));
  using DQIndices = decltype(outputOf(dqr.q));
  using DRIndices = decltype(mappedIndices(outputOf(dqr.r), DRMap{}));
  using DQRNodes = decltype(nodesOf(DQRMergeResult{}));
  auto q = Graph<QIndices,QRNodes>{};
  auto r = Graph<RIndices,QRNodes>{};

  Function<
    Graph<QR<QIndices,RIndices>,QRNodes>,
    Graph<QR<DQIndices,DRIndices>,DQRNodes>
  > f;

  f.set(a,a_val);
  f.evaluate();
  Mat33f q_val = f.get(q);
  Mat33f r_val = f.get(r);
  auto expected_qr = qrDecomposition(a_val);
  assert(q_val == expected_qr.q);
  assert(r_val == expected_qr.r);
  f.set(dq, dq_val);
  f.set(dr, dr_val);
  f.evaluateDerivs();

  Mat33f da = f.getDeriv(a);
  float h = 4e-3;

  // Verify the derivatives.
  for (size_t i=0; i!=3; ++i) {
    for (size_t j=0; j!=3; ++j) {
      float sum = 0;

      for (size_t i2=0; i2!=3; ++i2) {
        for (size_t j2=0; j2!=3; ++j2) {
          auto q_a =
            [&,i2,j2]{
              Mat33f q_val = qrDecomposition(a_val).q;
              return q_val.values[i2][j2];
            };

          sum += finiteDeriv(q_a, a_val.values[i][j], h) * dq_val.values[i2][j2];
        }
      }

      for (size_t i2=0; i2!=3; ++i2) {
        for (size_t j2=0; j2!=3; ++j2) {
          auto r_a =
            [&,i2,j2]{
              Mat33f r_val = qrDecomposition(a_val).r;
              return r_val.values[i2][j2];
            };

          sum += finiteDeriv(r_a, a_val.values[i][j], h) * dr_val.values[i2][j2];
        }
      }

      // da.values[0][0] is 0 but sum is 1
      assertNear(da.values[i][j],sum,5e-3);
    }
  }
}
#endif


int main()
{
  testFindNodeIndex();
  {
    auto a = var<struct A>();
    auto b = a+a;
    assert(eval(b,let(a,3)) == 6);
  }
  {
    auto a = 1_t;
    auto b = 2_t;
    auto c = a+b;
    assert(eval(c) == 3);
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
    auto a = var<struct A>();
    auto b = var<struct B>();
    auto c = a/b;
    assert(eval(c,let(a,3.0f),let(b,4.0f)) == 3.0f/4.0f);
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
    auto a = mat33Var<struct A>();
    auto aT0 = col<0>(a);
    Vec3f row0 = vec3(1,2,3);
    Vec3f row1 = vec3(4,5,6);
    Vec3f row2 = vec3(7,8,9);
    auto result = eval(aT0,let(a,mat33(row0,row1,row2)));
    auto expected_result = vec3(xValue(row0),xValue(row1),xValue(row2));
    assert(result == expected_result);
  }
  {
    auto vx = var<struct VX>();
    auto vy = var<struct VY>();
    auto vz = var<struct VZ>();
    auto v = vec3(vx,vy,vz);
    float result = eval(mag(v),let(vx,1),let(vy,2),let(vz,3));
    float expected_result = mag(vec3(1,2,3));
    assert(result == expected_result);
  }
#if ADD_QR_DECOMP
  {
    using RandomEngine = std::mt19937;
    RandomEngine engine(/*seed*/1);
    Mat33f a_value = randomMat33(engine);
    auto a = var<struct A>();
    auto qr = qrDecomposition(a);
    auto q_result = eval(qr.q,let(a,a_value));
    auto r_result = eval(qr.r,let(a,a_value));
    auto qr_value = qrDecomposition(a_value);
    auto expected_q_result = qr_value.q;
    auto expected_r_result = qr_value.r;
    assert(q_result == expected_q_result);
    assert(r_result == expected_r_result);
  }
#endif
  testFindAdjoint();
  testAddDeriv();
  testMulFunction();
  testDotFunction();
  testDivFunction();
  testSqrtFunction();
  testMagFunction();
  test3();
#if ADD_QR_DECOMP
  testQRDecompFunction();
#endif
}
