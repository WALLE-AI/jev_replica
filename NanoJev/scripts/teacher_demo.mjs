import { parseArgs } from 'node:util';
import { evaluateTeacher, MAX_OUTPUT_TOKENS, TeacherError } from './teachers.mjs';

// 仅一个自写案例；默认 dry-run。由调用者在外部配置凭据，本脚本不读取 .env。
// live 用法：node scripts/teacher_demo.mjs --live --teacher llm --model 提供方/模型
const state = {
  message: '同一笔订单扣款两次，请退还重复扣的钱。客服已受理，但退款尚未完成。',
  service_status: '可以正常登录和使用服务，没有报告紧急中断。',
};
const questions = {
  refunded: { type: 'boolean', instructions: '是否已经完成退款？提出申请或承诺不算完成。' },
  team: {
    type: 'choice', instructions: '应由哪个支持团队处理？',
    criteria: { billing: '扣费、支付和退款问题。', technical: '登录失败或服务技术故障。' },
  },
  severity: {
    type: 'score', instructions: '根据明确陈述的影响，判断问题的紧急程度。',
    criteria: ['常规请求，未描述服务受阻。', '部分功能受阻，但有替代方法。', '关键服务完全中断，没有替代方法。'],
  },
};

async function main() {
  let values;
  try {
    ({ values } = parseArgs({
      options: {
        live: { type: 'boolean', default: false },
        teacher: { type: 'string', default: 'llm' },
        model: { type: 'string' },
        help: { type: 'boolean', default: false },
      },
      strict: true,
      allowPositionals: false,
    }));
  } catch {
    throw new TeacherError('INVALID_ARGUMENTS', '命令行参数无效；使用 --help 查看用法。');
  }
  if (values.help) {
    console.log('默认离线：node scripts/teacher_demo.mjs\n单次调用：node scripts/teacher_demo.mjs --live --teacher llm --model 提供方/模型\n可替换为 --teacher jev --model typesafe-ai/jev。脚本不读取 .env，凭据由调用者配置。');
    return;
  }
  if (!['jev', 'llm'].includes(values.teacher)) throw new TeacherError('INVALID_ARGUMENTS', 'teacher 只能是 jev 或 llm。');
  if (values.live && !values.model?.trim()) throw new TeacherError('INVALID_ARGUMENTS', 'live 调用必须显式提供 --model。');
  if (!values.live) {
    console.log(JSON.stringify({
      mode: 'dry-run', teacher: values.teacher, model: values.model ?? null,
      requests: 1, maxRetries: 0,
      maxOutputTokens: values.teacher === 'llm' ? MAX_OUTPUT_TOKENS : null,
      target_kind: values.teacher === 'llm' ? 'hard_labels' : 'native_probabilities_or_rounded_probabilities',
      state, questions,
    }, null, 2));
    return;
  }
  const controller = new AbortController();
  const cancel = () => controller.abort();
  process.once('SIGINT', cancel);
  try {
    const result = await evaluateTeacher({
      teacher: values.teacher, model: values.model, state, questions, signal: controller.signal,
    });
    console.log(JSON.stringify(result, null, 2));
  } finally {
    process.removeListener('SIGINT', cancel);
  }
}

main().catch(error => {
  const safe = error instanceof TeacherError;
  console.error(JSON.stringify({
    error: safe ? error.code : 'DEMO_FAILED',
    message: safe ? error.message : '演示失败；未输出原始错误详情。',
    ...(safe && error.usage !== undefined ? { usage: error.usage } : {}),
  }));
  process.exitCode = 1;
});
