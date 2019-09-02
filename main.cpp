#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>
#include <type_traits>

#define ADD_QR_DECOMP 0
#define ADD_TEST 0


using std::cerr;
using std::is_same_v;


namespace {

template <typename Tag> struct Var { };
template <typename ValueType> struct Const { };
struct External {};
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


template <size_t expr_index,typename Nodes> struct Graph
{
  operator float() const;
};



template <size_t expr_index,typename Nodes>
constexpr auto indexOf(Graph<expr_index,Nodes>)
{
  return Indexed<expr_index>{};
}


template <size_t expr_index,typename Nodes>
Nodes nodesOf(Graph<expr_index,Nodes>);


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

template <typename First,size_t node_index,size_t adjoint_node_index>
struct AdjointList : First, MapEntry<node_index,adjoint_node_index> {
};

}


template <typename First,size_t node_index,size_t adjoint_node_index>
static constexpr size_t
  findAdjoint(
    AdjointList<First,node_index,adjoint_node_index>,
    Indexed<node_index>
  )
{
  return adjoint_node_index;
}


template <
  typename First,
  size_t node_index1,
  size_t node_index2,
  size_t adjoint_node_index
>
static constexpr size_t
  findAdjoint(
    AdjointList<First,node_index1,adjoint_node_index>,
    Indexed<node_index2>
  )
{
  return findAdjoint(First{},Indexed<node_index2>{});
}


template <typename Adjoints,size_t index>
static constexpr auto find_adjoint = findAdjoint(Adjoints{},Indexed<index>{});


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
template <typename Tag,typename T>
struct Dual {
  T &value;
};
}


namespace {
template <size_t from_index,size_t to_index>
constexpr auto findNewIndex(MapEntry<from_index,to_index>)
{
  return Indexed<to_index>{};
}
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
  return Graph<0,List<Node0>>{};
}
}


namespace {
template <typename Tag>
auto constant()
{
  using Expr = Const<Tag>;
  using Node0 = Node<0,Expr>;
  return Graph<0,List<Node0>>{};
}
}


template <typename Value>
static float
  floatValue(
    Graph<0,
      List<
        Node<0,Const<Value>>
      >
    >
  )
{
  return Value::value();
}


template <size_t expr_index,typename Nodes>
Graph<expr_index,Nodes>::operator float() const
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
template <typename First,typename Entry>
struct MapList : First, Entry {
};
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
template <typename... NodesA,typename...NodesB>
auto merge(List<NodesA...>,List<NodesB...>)
{
  using MergedNodes = List<NodesA...>;
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
  using MapB        = typename MergeResult::MapB;
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
auto operator-(Graph<index_a,NodesA> graph_a,Graph<index_b,NodesB> graph_b)
{
  return binary<Sub>(graph_a,graph_b);
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
template <
  size_t index_a,typename NodesA,
  size_t index_b,typename NodesB
>
auto operator/(Graph<index_a,NodesA> graph_a,Graph<index_b,NodesB> graph_b)
{
  return binary<Div>(graph_a,graph_b);
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
template <typename Tag,typename T>
auto dual(Graph<0,List<Node<0,Var<Tag>>>>,T& value)
{
  return Dual<Tag,T>{value};
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
struct Vec3f {
  const float x,y,z;

  bool operator==(const Vec3f &arg) const
  {
    return x==arg.x && y==arg.y && z==arg.z;
  }
};
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


namespace {


template <typename Row0,typename Row1,typename Row2,typename Nodes>
struct Mat33
{
  static Mat33f eval(const Vec3f &row0,const Vec3f &row1,const Vec3f &row2)
  {
    return mat33(row0,row1,row2);
  }
};


template <size_t x,size_t y,size_t z> struct Mat33Row;


}


namespace {
template <size_t x_index,size_t y_index,size_t z_index,typename Nodes>
struct Vec3
{
  static Vec3f eval(float x,float y,float z) { return Vec3f{x,y,z}; }
};
}


template <
  size_t m00,size_t m01,size_t m02,typename Row1,typename Row2,typename Nodes
>
static auto row(const Mat33<Mat33Row<m00,m01,m02>,Row1,Row2,Nodes>,Indexed<0>)
{
  return Vec3<m00,m01,m02,Nodes>{};
}


static Vec3f vec3(float x,float y,float z)
{
  return Vec3f{x,y,z};
}


template <size_t index>
static auto row(const Mat33f &m,Indexed<index>)
{
  float x = m.values[index][0];
  float y = m.values[index][1];
  float z = m.values[index][2];
  return vec3(x,y,z);
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
auto operator-(Vec3<xa,ya,za,NodesA> a,Vec3<xb,yb,zb,NodesB> b)
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
auto operator*(Vec3<x,y,z,NodesA> a,Graph<index_b,NodesB> b)
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
auto operator/(Vec3<x,y,z,NodesA> a,Graph<index_b,NodesB> b)
{
  auto new_x = xValue(a)/b;
  auto new_y = yValue(a)/b;
  auto new_z = zValue(a)/b;

  return vec3(new_x,new_y,new_z);
}
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
template <typename A,typename Values,typename Lets>
auto evalExpr(Const<A>, const Values &,const Lets &)
{
  return A::value();
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
    Graph<index, Nodes>,
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
    Vec3<x_index,y_index,z_index,Nodes>,
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
auto get(Graph<index,Nodes>,const Values &values)
{
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
  return Vec3<new_x_index,new_y_index,new_z_index,XYZNodes>{};
}


namespace {
template <typename Row0,typename Row1,typename Row2> struct Mat33Indices { };
}


namespace {
template <typename Tag>
auto
  let(
    Mat33<
      Mat33Row<0,1,2>,
      Mat33Row<3,4,5>,
      Mat33Row<6,7,8>,
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
    >,
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
  return Mat33<
    Mat33Row<0,1,2>,
    Mat33Row<3,4,5>,
    Mat33Row<6,7,8>,
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
    Mat33<
      Mat33Row<r0xi,r0yi,r0zi>,
      Mat33Row<r1xi,r1yi,r1zi>,
      Mat33Row<r2xi,r2yi,r2zi>,
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
  size_t r0x,size_t r0y,size_t r0z,
  size_t r1x,size_t r1y,size_t r1z,
  size_t r2x,size_t r2y,size_t r2z,
  typename Row0Nodes,typename Row1Nodes,typename Row2Nodes
>
static auto mat33(
  Vec3<r0x,r0y,r0z,Row0Nodes>,
  Vec3<r1x,r1y,r1z,Row1Nodes>,
  Vec3<r2x,r2y,r2z,Row2Nodes>
)
{
  using Row01MergeResult = decltype(merge(Row0Nodes{},Row1Nodes{}));
  using Row01Nodes = typename Row01MergeResult::Nodes;
  using MapRow1 = typename Row01MergeResult::MapB;
  using Row012MergeResult = decltype(merge(Row01Nodes{},Row2Nodes{}));
  using Row012Nodes = typename Row012MergeResult::Nodes;
  using MapRow2 = typename Row012MergeResult::MapB;
  constexpr size_t new_r0x = r0x;
  constexpr size_t new_r0y = r0y;
  constexpr size_t new_r0z = r0z;
  constexpr size_t new_r1x = mapped_index<r1x,MapRow1>;
  constexpr size_t new_r1y = mapped_index<r1y,MapRow1>;
  constexpr size_t new_r1z = mapped_index<r1z,MapRow1>;
  constexpr size_t new_r2x = mapped_index<r2x,MapRow2>;
  constexpr size_t new_r2y = mapped_index<r2y,MapRow2>;
  constexpr size_t new_r2z = mapped_index<r2z,MapRow2>;

  return
    Mat33<
      Mat33Row<new_r0x,new_r0y,new_r0z>,
      Mat33Row<new_r1x,new_r1y,new_r1z>,
      Mat33Row<new_r2x,new_r2y,new_r2z>,
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
  return mat33(
    vec3(xValue(col0),xValue(col1),xValue(col2)),
    vec3(yValue(col0),yValue(col1),yValue(col2)),
    vec3(zValue(col0),zValue(col1),zValue(col2))
  );
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto xValue(Vec3<x,y,z,Nodes>)
{
  return Graph<x,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto yValue(Vec3<x,y,z,Nodes>)
{
  return Graph<y,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto zValue(Vec3<x,y,z,Nodes>)
{
  return Graph<z,Nodes>{};
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


template <
  size_t m10,size_t m11,size_t m12,typename Row0,typename Row2,typename Nodes
>
static auto
  row(const Mat33<Row0,Mat33Row<m10,m11,m12>,Row2,Nodes>,Indexed<1>)
{
  return Vec3<m10,m11,m12,Nodes>{};
}


template <
  size_t m20,size_t m21,size_t m22,typename Row0,typename Row1,typename Nodes
>
static auto
  row(const Mat33<Row0,Row1,Mat33Row<m20,m21,m22>,Nodes>,Indexed<2>)
{
  return Vec3<m20,m21,m22,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Vec3<x,y,z,Nodes>,Indexed<0>)
{
  return Graph<x,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Vec3<x,y,z,Nodes>,Indexed<1>)
{
  return Graph<y,Nodes>{};
}


template <size_t x,size_t y,size_t z,typename Nodes>
static auto elem(Vec3<x,y,z,Nodes>,Indexed<2>)
{
  return Graph<z,Nodes>{};
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
static auto sqrt(Graph<index,Nodes>)
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
  auto zero = constant<Zero>();
  auto row0 = vec3( r11, r12,r13);
  auto row1 = vec3(zero, r22,r23);
  auto row2 = vec3(zero,zero,r33);
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
  return AdjointList<OldAdjointList,adjoint_index,value_index>{};
}


// adjoints[i] += k
template <typename Adjoints,typename... Nodes,size_t i,size_t k>
static auto addTo(AdjointGraph<Adjoints,List<Nodes...>>,Indexed<i>,Indexed<k>)
{
  constexpr size_t adjoint_i = find_adjoint<Adjoints,i>;

  using InsertResult =
    decltype(insertNode(List<Nodes...>{},Add<adjoint_i,k>{}));

  using NewNodes = decltype(newNodesOf(InsertResult{}));
  constexpr size_t new_index = newIndexOf(InsertResult{});

  using NewAdjoints =
    decltype(setAdjoint(Adjoints{},Indexed<i>{},Indexed<new_index>{}));

  return AdjointGraph<NewAdjoints,NewNodes>{};
}


template <typename Adjoints,typename Nodes,size_t adjoint_k,size_t i,size_t j>
static auto
  addMul(AdjointGraph<Adjoints,Nodes>,Indexed<i>,Indexed<adjoint_k>,Indexed<j>)
{
  // adjoints[i] += adjoints[k]*j;
  //

  // Insert adjoints[k]*j into the nodes
  using InsertResult1 =
    decltype(insertNode(Nodes{},Mul<adjoint_k,j>{}));
    // This is a UpdateMergedNodesResult

  using NewNodes1 = decltype(newNodesOf(InsertResult1{}));
  constexpr size_t ak_times_aj = newIndexOf(InsertResult1{});

  // add adjoints[k]*j to adjoints[i]
  using NewGraph1 =
    decltype(
      addTo(
        AdjointGraph<Adjoints,NewNodes1>{},
        Indexed<i>{},
        Indexed<ak_times_aj>{}
      )
    );

  using Adjoints1 = decltype(adjointsOf(NewGraph1{}));
  using NewNodes2 = decltype(nodesOf(NewGraph1{}));

  return AdjointGraph<Adjoints1,NewNodes2>{};
}


template <typename AdjointGraph,size_t k,typename Tag>
static auto addDeriv(AdjointGraph,Node<k,Var<Tag>>)
{
  return AdjointGraph{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i,size_t j>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Add<i,j>>)
{
  // adjoints[i] += adjoints[k];
  // adjoints[j] += adjoints[k];
  //
  constexpr size_t adjoint_k = find_adjoint<Adjoints,k>;

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
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,Mul<i,j>>)
{
  constexpr size_t adjoint_k = find_adjoint<Adjoints,k>;

  // adjoints[i] += adjoints[k]*j
  using NewGraph1 =
    decltype(
      addMul(
        AdjointGraph<Adjoints,Nodes>{},
        Indexed<i>{},
        Indexed<adjoint_k>{},
        Indexed<j>{}
      )
    );

  // adjoints[j] += adjoints[k]*i;
  using NewGraph2 =
    decltype(
      addMul(
        NewGraph1{},
        Indexed<j>{},
        Indexed<adjoint_k>{},
        Indexed<i>{}
      )
    );

  return NewGraph2{};
}


template <typename Adjoints,typename Nodes,size_t k,size_t i>
static auto addDeriv(AdjointGraph<Adjoints,Nodes>,Node<k,XValue<i>>)
{
  constexpr size_t adjoint_k = find_adjoint<Adjoints,k>;

  // adjoints[i].x += adjoints[k]

  return
    addTo(AdjointGraph<Adjoints,Nodes>{},Indexed<i>{},Indexed<adjoint_k>{});
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


template <size_t zero_node>
static auto makeZeroAdjoints(Indexed<0>,Indexed<zero_node>)
{
  return Empty{};
}


template <size_t node_count,size_t zero_node>
static auto
  makeZeroAdjoints(Indexed<node_count>,Indexed<zero_node>)
{
  using First =
    decltype(makeZeroAdjoints(Indexed<node_count-1>{},Indexed<zero_node>{}));

  return AdjointList<First,node_count-1,zero_node>{};
}


template <typename Adjoints,typename Nodes,size_t result_index>
static auto
  addAdjoint(AdjointGraph<Adjoints,Nodes>,Indexed<result_index>)
{
  // Add a node that we'll use for storing the derivative of the result.
  constexpr size_t new_dresult_index = listSize<Nodes>;
  using NewNodes = decltype(addNode(Nodes{},External{}));

  // Set the adjoint of our result node
  using NewAdjoints =
    decltype(
      setAdjoint(
        Adjoints{},
        /*adjoint_node*/Indexed<result_index>{},
        /*value*/Indexed<new_dresult_index>{}
      )
    );

  return AdjointGraph<NewAdjoints,NewNodes>{};
}


template <typename Adjoints,typename Nodes>
static auto addAdjoints(AdjointGraph<Adjoints,Nodes>,Indices<>)
{
  return AdjointGraph<Adjoints,Nodes>{};
}


template <
  typename Adjoints,
  typename Nodes,
  size_t first_result_index,
  size_t... rest_result_indices
>
static auto
  addAdjoints(
    AdjointGraph<Adjoints,Nodes>,
    Indices<first_result_index,rest_result_indices...>
  )
{
  using AdjointGraph2 =
    decltype(
      addAdjoint(
        AdjointGraph<Adjoints,Nodes>{},
        Indexed<first_result_index>{}
      )
    );

  return addAdjoints(AdjointGraph2{},Indices<rest_result_indices...>{});
}


// Create the nodes that contain the adjoints by going through a forward
// and reverse pass.  In the forward pass, we introduce an adjoint node
// for each node.  In the reverse pass, we update the adjoint nodes.
template <
  typename... Nodes,
  typename ResultIndices
>
static auto adjointNodes(ResultIndices,List<Nodes...>)
{
  // Add a zero node to the nodes if we don't have one, since we'll need
  // to initialize all the adjoints to this.
  using InsertResult = decltype(insertNode(List<Nodes...>{},Const<Zero>{}));
  using NodesWithZero = decltype(newNodesOf(InsertResult{}));
  constexpr size_t zero_index = newIndexOf(InsertResult{});

  // Build the initial set of adjoints where all nodes are zero.
  using Adjoints =
    decltype(
      makeZeroAdjoints(Indexed<sizeof...(Nodes)>{},Indexed<zero_index>{})
    );

  using AdjointGraph2 =
    decltype(
      addAdjoints(
        AdjointGraph<Adjoints,NodesWithZero>{},
        ResultIndices{}
      )
    );

  using Adjoints2 = decltype(adjointsOf(AdjointGraph2{}));
  using NodesWithDResult = decltype(nodesOf(AdjointGraph2{}));

  // Process the nodes in reverse, adding new nodes and updating the adjoints.
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


template <
  typename... Nodes1,
  size_t result_index
    // I think this needs to be generalized to take a list of result
    // indices.
>
static auto adjointNodes(Indexed<result_index>,List<Nodes1...>)
{
  return adjointNodes(Indices<result_index>{},List<Nodes1...>{});
}



static void testFindAdjoint()
{
  using Adjoints =
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<
    AdjointList<Empty,
    0, 5>,
    1, 5>,
    2, 5>,
    3, 5>,
    4, 5>,
    4, 6>,
    2, 8>,
    3, 10>,
    0, 12>, // dA = C*B
    1, 14>;

  constexpr size_t result = findAdjoint(Adjoints{},Indexed<0>{});
  static_assert(result == 12);
}


static void testMakeZeroAdjoints()
{
  constexpr size_t zero_node = 1;
  constexpr size_t node_count = 2;

  using Result =
    decltype(
      makeZeroAdjoints(
        Indexed<node_count>{},
        Indexed<zero_node>{}
      )
    );

  using Expected =
    AdjointList<
      AdjointList<
        Empty, 0, zero_node
      >,
      1, zero_node
    >;

  static_assert(is_same_v<Result,Expected>);
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
    AdjointList<
      AdjointList<
        AdjointList<Empty,0,3>,  // adjoints[0] = 3
        1,3                      // adjoints[1] = 3
      >,
      2,4                        // adjoints[2] = 4
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
    AdjointList<
      AdjointList<
        AdjointList<
          AdjointList<
            AdjointList<Empty,0,3>,  // adjoints[0] = 3
            1,3                      // adjoints[1] = 3
          >,
          2,4                        // adjoints[2] = 4
        >,
        0,6                          // adjoints[0] = 6 // 0 + 1*B
      >,
      1,8                            // adjoints[1] = 8 // 0 + 1*A
    >;

  using Expected = AdjointGraph<ExpectedAdjoints,ExpectedNodes>;
  static_assert(is_same_v<Result,Expected>);
}


template <size_t index,typename... Nodes>
static auto nodesOf(Graph<index,List<Nodes...>>)
{
  return List<Nodes...>{};
}


// This needs to get the variable tag and look that up in the adjoint
// grpah.
template <typename AdjointGraph,typename Tag>
static auto
  varAdjoint(
    Graph<
      0,
      List<
        Node<0,Var<Tag>>
      >
    >,
    AdjointGraph
  )
{
  using NodeIndex =
    decltype(findNodeIndex(Var<Tag>{},nodesOf(AdjointGraph{})));

  return Indexed<findAdjoint(adjointsOf(AdjointGraph{}),NodeIndex{})>{};
}


template <typename Variable,typename Values,typename AdjointGraph>
static auto adjointValue(Variable,const Values &values,AdjointGraph)
{
  using Index = decltype(varAdjoint(Variable{},AdjointGraph{}));
  return getValue(Index{},values);
}


template <typename Variable,size_t n_values,typename AdjointGraph>
static auto adjointValue(Variable,const float (&values)[n_values],AdjointGraph)
{
  using Index = decltype(varAdjoint(Variable{},AdjointGraph{}));
  assert(Index::value < n_values);
  return values[Index::value];
}


template <typename Graph,size_t n_nodes,typename AdjointNodes>
static void
  setValue(Graph,float value,float (&values)[n_nodes],AdjointNodes)
{
  constexpr size_t index = decltype(indexOf(Graph{}))::value;
  using Nodes = decltype(nodesOf(Graph{}));
  using MergeResult = decltype(merge(AdjointNodes{},Nodes{}));
  using Map = typename MergeResult::MapB;
  constexpr size_t mi = mapped_index<index,Map>;
  values[mi] = value;
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
  evaluate<begin+1,end>(List<Nodes...>{},values);
}

}



static void testAdjointNodes()
{
  // Define our graph
  auto a = var<struct A>();
  auto b = var<struct B>();
  auto c = var<struct C>();
  auto graph = a*b*c;
  using AdjointGraph = decltype(adjointNodes(indexOf(graph),nodesOf(graph)));
  using Adjoints = decltype(adjointsOf(AdjointGraph{}));

  float a_val = 5;
  float b_val = 6;
  float c_val = 7;
  float dgraph_val = 1;
  using NewNodes = decltype(nodesOf(AdjointGraph{}));
  using Adjoints = decltype(adjointsOf(AdjointGraph{}));
  static constexpr size_t graph_index = decltype(indexOf(graph))::value;

  constexpr size_t dgraph_index =
    findAdjoint(Adjoints{},Indexed<graph_index>{});

  static constexpr size_t n_values = sizeOf(NewNodes{});
  float values[n_values];

  // Set the inputs
  setValue(a,a_val,values,NewNodes{});
  setValue(b,b_val,values,NewNodes{});
  setValue(c,c_val,values,NewNodes{});

  // Evaluate the normal part of the graph
  evaluate<0,dgraph_index>(NewNodes{},values);

  // Verify the output
  assert(values[graph_index] == a_val*b_val*c_val);

  // Set the derivative of the output
  values[dgraph_index] = dgraph_val;

  // Evaluate the rest of the graph
  evaluate<dgraph_index+1,n_values>(NewNodes{},values);

  // Extract the derivatives
  float da_val = adjointValue(a,values,AdjointGraph{});
  float db_val = adjointValue(b,values,AdjointGraph{});
  float dc_val = adjointValue(c,values,AdjointGraph{});

  // Verify
  assert(da_val == b_val*c_val);
  assert(db_val == a_val*c_val);
  assert(dc_val == a_val*b_val);
}


static void testDotAdjointNodes()
{
  // Define our graph
  auto ax = var<struct AX>();
  auto ay = var<struct AY>();
  auto az = var<struct AZ>();
  auto bx = var<struct BX>();
  auto by = var<struct BY>();
  auto bz = var<struct BZ>();
  auto a = vec3(ax,ay,az);
  auto b = vec3(bx,by,bz);
  auto graph = dot(a,b);
  constexpr size_t graph_index = decltype(indexOf(graph))::value;

  using AdjointGraph =
    decltype(adjointNodes(Indexed<graph_index>{},nodesOf(graph)));

  // Evaluate the adjoint graph
  auto a_val = vec3(1,2,3);
  auto b_val = vec3(4,5,6);
  float dgraph_val = 1;
  using NewNodes = decltype(nodesOf(AdjointGraph{}));
  using Adjoints = decltype(adjointsOf(AdjointGraph{}));
  static constexpr size_t n_values = sizeOf(NewNodes{});
  float values[n_values];
  setValue(ax,a_val.x,values,NewNodes{});
  setValue(ay,a_val.y,values,NewNodes{});
  setValue(az,a_val.z,values,NewNodes{});
  setValue(bx,b_val.x,values,NewNodes{});
  setValue(by,b_val.y,values,NewNodes{});
  setValue(bz,b_val.z,values,NewNodes{});

  constexpr size_t dgraph_index =
    findAdjoint(Adjoints{},Indexed<graph_index>{});

  evaluate<0,dgraph_index>(NewNodes{},values);
  values[dgraph_index] = dgraph_val;
  evaluate<dgraph_index+1,n_values>(NewNodes{},values);

  // Extract the derivatives
  float dax_val = adjointValue(ax,values,AdjointGraph{});
  float day_val = adjointValue(ay,values,AdjointGraph{});
  float daz_val = adjointValue(az,values,AdjointGraph{});
  float dbx_val = adjointValue(bx,values,AdjointGraph{});
  float dby_val = adjointValue(by,values,AdjointGraph{});
  float dbz_val = adjointValue(bz,values,AdjointGraph{});
  Vec3f da_val = vec3(dax_val,day_val,daz_val);
  Vec3f db_val = vec3(dbx_val,dby_val,dbz_val);

  // Verify
  assert(da_val == vec3(4,5,6));
  assert(db_val == vec3(1,2,3));
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


template <size_t n,size_t index>
void setValue(float (&)[n],Node<index,External>)
{
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Mul<a,b>>)
{
  values[index] = values[a] * values[b];
}


template <size_t n,size_t index,size_t a,size_t b>
void setValue(float (&values)[n],Node<index,Add<a,b>>)
{
  values[index] = values[a] + values[b];
}

}


namespace {
template <typename Graph> struct Function;

template <size_t value_index,typename ValueNodes>
struct Function< Graph<value_index,ValueNodes> >
{
  using ValueGraph = Graph<value_index,ValueNodes>;

  using AdjointGraph =
    decltype(
      adjointNodes(
        Indices<value_index>{},
        ValueNodes{}
      )
    );

  using AdjointNodes = decltype(nodesOf(AdjointGraph{}));
  using Adjoints = decltype(adjointsOf(AdjointGraph{}));

  template <size_t index,typename Nodes>
  static constexpr size_t mappedIndex(Graph<index,Nodes>)
  {
    using MergeResult = decltype(merge(AdjointNodes{},Nodes{}));
    using Map = typename MergeResult::MapB;
    return mapped_index<index,Map>;
  }

  template <typename G>
  static constexpr size_t derivIndex(G)
  {
    constexpr size_t index = mappedIndex(G{});
    return findAdjoint(Adjoints{},Indexed<index>{});
  }

  static constexpr size_t dresult_index = derivIndex(ValueGraph{});
  static constexpr size_t n_values = sizeOf(AdjointNodes{});
  float values[n_values];

  template <typename Graph,typename T>
  void set(Graph,const T& value)
  {
    setValue(Graph{},value,values,AdjointNodes{});
  }

  void evaluate()
  {
    constexpr size_t n_value_nodes = listSize<ValueNodes>;
    ::evaluate<0,n_value_nodes>(AdjointNodes{},values);
  }

  template <typename Graph>
  auto get(Graph) const
  {
    return values[mappedIndex(Graph{})];
  }

  template <typename Graph,typename T>
  void setDeriv(Graph,const T& value)
  {
    values[derivIndex(Graph{})] = value;
  }

  void evaluateDerivs()
  {
    constexpr size_t n_value_nodes = listSize<ValueNodes>;
    ::evaluate<n_value_nodes,n_values>(AdjointNodes(),values);
  }

  template <typename Graph>
  float getDeriv(Graph) const
  {
    return values[derivIndex(Graph{})];
  }
};
}



static void testMulFunction()
{
  auto a = var<struct A>();
  auto b = var<struct B>();
  auto c = a*b;
  float a_val = 5;
  float b_val = 6;
  Function< decltype(c) > f;
  f.set(a,a_val);
  f.set(b,b_val);
  f.evaluate();
  assert(f.get(c) == a_val*b_val);
  f.setDeriv(c,1);
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
  Function< decltype(c) > f;
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


  f.setDeriv(c,1);
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


#if ADD_TEST
static void testQRDecompFunction()
{
  std::mt19937 engine(/*seed*/1);
  auto a = mat33Var<struct A>();
  auto qr = qrDecomposition(a);
  Mat33f dr = randomMat33(engine);
  Mat33f dq = randomMat33(engine);
  auto dqr = ::qr(dq,dr);

  // qr is a QR<Q,R>, where Q and R are graphs with their own nodes.
  // It needs to be that way so that the qrDecomposition() function is
  // agnostic to the representation of the result.
  using MergeResult = decltype(merge(nodesOf(qr.q),nodesOf(qr.r)));
  using MergedNodes = decltype(nodesOf(MergeResult{}));
  using RMap = decltype(mapOf(MergeResult{}));
  using MappedR = decltype(map(qr.r, RMap{}));
  Function< decltype(nodesIndicesOf(qr)), decltype(nodesOf(qr)) > f;
  Mat33f a_val = mat33(vec3(1,2,3),vec3(4,5,6),vec(7,8,9));

  f.set(a,a_val);
  f.evaluate();
  auto qr_val = f.get(qr);
  assert(qr_val == qrDecomposition(a_val));
  f.setDeriv(qr,dqr);
  f.evaluateDerivs();
  Mat33f da = f.getDeriv(a);

  // Verify that the derivatives are correct.
  assert(false);

  // da is d(error)/d(a) given d(error)/d(qr)
  // How do we verify this?
  // fda[i][j] = sum{i2,j2}(d(qr[i2][j2])/d(a[i][j])*dqr[i2][j2])
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
  testMakeZeroAdjoints();
  testAddDeriv();
  testAdjointNodes();
  testDotAdjointNodes();
  testMulFunction();
  testDotFunction();
#if ADD_TEST
  testQRDecompFunction();
#endif
}
